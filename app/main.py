"""FastAPI web app: upload reports and review exhibit dose verdicts."""

from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import assess
from .csvio import sample_doses_csv, sample_placements_csv
from .db import Store
from .errors import ReportError
from .service import ReportService
from .timeutil import format_hour, format_iso, format_minute

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR.parent / "data" / "exhibits.sqlite3"


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path or os.environ.get("EXHIBIT_DB", str(DEFAULT_DB))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    store = Store(path)
    store.init_schema()
    service = ReportService(store)

    # Queued submissions block on the order gate while waiting for their
    # turn.  Keep them on a dedicated executor so they can never occupy
    # every shared thread and starve the roster pages.
    submit_executor = ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="submit"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        submit_executor.shutdown(wait=True)

    app = FastAPI(title="展柜照度剂量复核", lifespan=lifespan)
    app.state.store = store
    app.state.service = service
    app.state.db_path = path
    app.state.submit_executor = submit_executor

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["minute"] = format_minute
    templates.env.filters["hour"] = format_hour
    templates.env.filters["iso"] = format_iso
    templates.env.filters["number"] = _fmt_number

    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, status: str | None = None, message: str | None = None):
        if status not in (None, "", *assess.STATUS_LABELS):
            status = None
        context = _index_context(service)
        context.update({"active_status": status or "", "message": message})
        if status:
            context["assessments"] = service.list_assessments(status)
        return templates.TemplateResponse(request, "index.html", context)

    @app.post("/submit")
    async def submit(
        request: Request,
        placements: UploadFile | None = File(None),
        doses: UploadFile | None = File(None),
        illuminance_cap: str = Form(""),
        dose_threshold: str = Form(""),
    ):
        try:
            cap = _optional_float(illuminance_cap, "照度上限")
            threshold = _optional_float(dose_threshold, "剂量阈值")
            placements_raw, placements_name = await _read_upload(placements)
            doses_raw, doses_name = await _read_upload(doses)
        except ReportError as exc:
            return _error_page(request, templates, service, exc.messages,
                               illuminance_cap, dose_threshold)
        except UnicodeDecodeError:
            return _error_page(
                request, templates, service,
                ["CSV 编码无法识别，请另存为 UTF-8 后重试"],
                illuminance_cap, dose_threshold,
            )

        # Take the queue ticket only once the whole form has arrived, so
        # replacements take effect in the order the service received their
        # complete forms — not in the order connections happened to open.
        ticket = service.gate.ticket()
        try:
            # The submission blocks on the order gate and on SQLite; run it
            # on the dedicated executor so queued reports never stall the
            # roster pages.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                submit_executor,
                functools.partial(
                    service.submit,
                    placements_raw=placements_raw,
                    placements_name=placements_name,
                    doses_raw=doses_raw,
                    doses_name=doses_name,
                    illuminance_cap=cap,
                    dose_threshold=threshold,
                    ticket=ticket,
                ),
            )
        except ReportError as exc:
            return _error_page(request, templates, service, exc.messages,
                               illuminance_cap, dose_threshold)
        except BaseException:
            # service.submit releases its own ticket on failure; this covers
            # the executor never getting to run it at all.
            service.gate.skip(ticket)
            raise

        labels = {"placements": "摆放表", "doses": "剂量表"}
        detail = "、".join(labels[kind] for kind in result.replaced)
        message = (
            f"{detail}换报成功：重算 {result.recomputed} 件展品，"
            f"撤下 {result.pruned} 条失效条目"
        )
        if result.missing_cabinet_hours:
            cabinets = sorted({c for c, _ in result.missing_cabinet_hours})
            message += (
                f"；有 {len(result.missing_cabinet_hours)} 个柜·小时缺测"
                f"（{', '.join(cabinets)}），相关结论按缺测处理"
            )
        return RedirectResponse(url=f"/?message={quote(message)}", status_code=303)

    @app.get("/exhibits/{exhibit_id}", response_class=HTMLResponse)
    def exhibit_detail(request: Request, exhibit_id: str):
        item = service.get_assessment(exhibit_id)
        if item is None:
            raise HTTPException(status_code=404, detail="名单中没有该展品")
        return templates.TemplateResponse(
            request, "detail.html", {"request": request, "exhibit": item}
        )

    @app.get("/samples/placements.csv")
    def sample_placements():
        return _csv_response("placements.csv", sample_placements_csv())

    @app.get("/samples/doses.csv")
    def sample_doses():
        return _csv_response("doses.csv", sample_doses_csv())

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def _index_context(service: ReportService) -> dict:
    return {
        "config": service.get_config(),
        "meta": service.report_meta(),
        "assessments": service.list_assessments(),
        "summary": service.status_summary(),
        "status_labels": assess.STATUS_LABELS,
        "active_status": "",
        "message": None,
        "errors": None,
        "form": None,
    }


def _error_page(
    request: Request,
    templates: Jinja2Templates,
    service: ReportService,
    errors: list[str],
    cap_text: str,
    threshold_text: str,
) -> HTMLResponse:
    context = _index_context(service)
    context.update(
        {
            "errors": errors,
            "form": {
                "illuminance_cap": cap_text,
                "dose_threshold": threshold_text,
            },
        }
    )
    return templates.TemplateResponse(
        request, "index.html", context, status_code=400
    )


def _optional_float(text: str, label: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ReportError(f"{label}“{text}”不是数字") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise ReportError(f"{label}必须是有限数字")
    return value


async def _read_upload(upload: UploadFile | None) -> tuple[bytes | None, str | None]:
    if upload is None or not upload.filename:
        return None, None
    raw = await upload.read()
    if not raw:
        return None, None
    return raw, upload.filename


def _csv_response(filename: str, text: str) -> Response:
    return Response(
        content=text.encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _fmt_number(value) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number == int(number):
        return f"{int(number):,}"
    return f"{number:,.2f}"
