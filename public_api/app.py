from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, REGISTRY, generate_latest

from bn_ml.config import load_config
from public_api.data_access import (
    build_models_archive,
    list_model_bundles,
    load_account_state,
    load_recent_trades,
    load_runtime_summary,
    load_training_status,
    read_prediction_snapshot,
)
from public_api.metrics import BNMLRuntimeCollector

HTTP_REQUESTS = Counter(
    "bnml_api_http_requests_total",
    "Total HTTP requests served by the BN-ML public API",
    labelnames=("method", "path", "status"),
)
HTTP_LATENCY = Histogram(
    "bnml_api_http_request_duration_seconds",
    "HTTP request duration in seconds for the BN-ML public API",
    labelnames=("method", "path"),
    buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.0, 5.0),
)


@dataclass(frozen=True)
class APISettings:
    config_path: Path
    db_path: Path
    models_dir: Path
    scan_path: Path
    opportunities_path: Path
    ws_poll_sec: float
    cors_origins: list[str]
    default_limit: int


def _resolve_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (cwd / path).resolve()


def _parse_origins(raw: Any) -> list[str]:
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
        return items or ["*"]
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return items or ["*"]
    return ["*"]


def _load_settings() -> APISettings:
    cwd = Path.cwd()
    config_path = Path(os.getenv("BNML_CONFIG_PATH", "configs/bot.yaml")).resolve()
    cfg = load_config(config_path)

    monitoring_cfg = cfg.get("monitoring", {})
    storage_cfg = cfg.get("storage", {})
    api_cfg = cfg.get("public_api", {})

    metrics_dir = _resolve_path(str(monitoring_cfg.get("metrics_dir", "artifacts/metrics")), cwd=cwd)
    db_path = _resolve_path(str(storage_cfg.get("sqlite_path", "artifacts/state/bn_ml.db")), cwd=cwd)
    models_dir = _resolve_path(
        str(os.getenv("BNML_MODELS_DIR", api_cfg.get("models_dir", "models"))),
        cwd=cwd,
    )

    ws_poll_sec = float(os.getenv("BNML_API_WS_POLL_SEC", api_cfg.get("ws_poll_sec", 2.0)))
    ws_poll_sec = max(0.5, min(ws_poll_sec, 10.0))
    default_limit = int(api_cfg.get("default_limit", 50))
    default_limit = max(1, min(default_limit, 500))

    cors_raw = os.getenv("BNML_API_CORS_ORIGINS")
    if cors_raw is None:
        cors_raw = api_cfg.get("cors_origins", ["*"])
    cors_origins = _parse_origins(cors_raw)

    return APISettings(
        config_path=config_path,
        db_path=db_path,
        models_dir=models_dir,
        scan_path=metrics_dir / "latest_scan.csv",
        opportunities_path=metrics_dir / "latest_opportunities.csv",
        ws_poll_sec=ws_poll_sec,
        cors_origins=cors_origins,
        default_limit=default_limit,
    )


def create_app() -> FastAPI:
    settings = _load_settings()
    app = FastAPI(
        title="BN-ML Public API",
        version="1.0.0",
        description="Public read-only API for BN-ML runtime predictions and monitoring.",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Register runtime collector once per process to enrich /metrics.
    try:
        REGISTRY.register(BNMLRuntimeCollector(db_path=settings.db_path, scan_path=settings.scan_path))
    except ValueError:
        pass

    site_dir = Path(__file__).resolve().parent / "site"
    app.mount("/assets", StaticFiles(directory=str(site_dir)), name="assets")

    @app.middleware("http")
    async def _metrics_middleware(request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = max(0.0, time.perf_counter() - start)
        path = request.url.path
        status = str(response.status_code)
        HTTP_REQUESTS.labels(request.method, path, status).inc()
        HTTP_LATENCY.labels(request.method, path).observe(elapsed)
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(site_dir / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(site_dir / "favicon.svg", media_type="image/svg+xml")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        snapshot = read_prediction_snapshot(settings.scan_path, limit=1)
        model_bundles = list_model_bundles(settings.models_dir)
        return {
            "status": "ok",
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(settings.config_path),
            "db_path": str(settings.db_path),
            "models_dir": str(settings.models_dir),
            "scan_path": str(settings.scan_path),
            "scan_available": bool(snapshot.get("total_rows", 0)),
            "scan_age_sec": snapshot.get("age_sec"),
            "db_available": settings.db_path.exists(),
            "models_available": settings.models_dir.exists(),
            "model_bundles": len(model_bundles),
        }

    @app.get("/api/v1/predictions")
    async def predictions(
        limit: int = Query(default=settings.default_limit, ge=1, le=500),
        signal: Optional[str] = Query(default=None),
    ) -> JSONResponse:
        payload = read_prediction_snapshot(settings.scan_path, limit=limit, signal=signal)
        payload["source"] = str(settings.scan_path)
        return JSONResponse(payload)

    @app.get("/api/v1/opportunities")
    async def opportunities(
        limit: int = Query(default=settings.default_limit, ge=1, le=500),
        signal: Optional[str] = Query(default=None),
    ) -> JSONResponse:
        payload = read_prediction_snapshot(settings.opportunities_path, limit=limit, signal=signal)
        payload["source"] = str(settings.opportunities_path)
        return JSONResponse(payload)

    @app.get("/api/v1/account")
    async def account() -> JSONResponse:
        payload = load_account_state(settings.db_path)
        return JSONResponse(payload)

    @app.get("/api/v1/trades")
    async def trades(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
        payload = load_recent_trades(settings.db_path, limit=limit)
        return JSONResponse({"total_rows": len(payload), "rows": payload})

    @app.get("/api/v1/summary")
    async def summary() -> JSONResponse:
        payload = load_runtime_summary(settings.db_path)
        payload["scan"] = read_prediction_snapshot(settings.scan_path, limit=1)
        return JSONResponse(payload)

    @app.get("/api/v1/training/status")
    async def training_status() -> JSONResponse:
        payload = load_training_status(settings.db_path)
        if not payload:
            payload = {
                "status": "idle",
                "phase": "waiting",
                "current_symbol": None,
                "progress_pct": 0.0,
                "symbols_queued": 0,
                "symbols_completed": 0,
                "symbols_trained": 0,
                "symbols_errors": 0,
            }
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/models")
    async def models() -> JSONResponse:
        bundles = list_model_bundles(settings.models_dir)
        return JSONResponse(
            {
                "models_dir": str(settings.models_dir),
                "total_bundles": len(bundles),
                "total_files": int(sum(int(item.get("file_count", 0)) for item in bundles)),
                "total_size_bytes": int(sum(int(item.get("size_bytes", 0)) for item in bundles)),
                "rows": bundles,
            }
        )

    @app.get("/api/v1/models/download")
    async def download_models(model_key: Optional[str] = Query(default=None)) -> StreamingResponse:
        try:
            filename, archive_bytes, file_count, total_bytes = build_models_archive(
                models_dir=settings.models_dir,
                model_key=model_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-BNML-Archive-Files": str(file_count),
            "X-BNML-Archive-Bytes": str(total_bytes),
        }
        return StreamingResponse(iter([archive_bytes]), media_type="application/zip", headers=headers)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/ws/predictions")
    async def ws_predictions(websocket: WebSocket) -> None:
        await websocket.accept()
        signal = websocket.query_params.get("signal")
        limit_raw = websocket.query_params.get("limit")
        try:
            limit = int(limit_raw) if limit_raw else settings.default_limit
        except ValueError:
            limit = settings.default_limit
        limit = max(1, min(limit, 500))

        last_generated_at: str | None = None
        try:
            while True:
                payload = read_prediction_snapshot(settings.scan_path, limit=limit, signal=signal)
                generated_at = payload.get("generated_at")
                if generated_at != last_generated_at:
                    await websocket.send_json({"type": "predictions", "payload": payload})
                    last_generated_at = generated_at
                await asyncio.sleep(settings.ws_poll_sec)
        except WebSocketDisconnect:
            return

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = _load_settings()
    host = os.getenv("BNML_API_HOST", "0.0.0.0")
    port = int(os.getenv("BNML_API_PORT", "8000"))
    uvicorn.run("public_api.app:app", host=host, port=port, reload=False)
