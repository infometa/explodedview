from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import BASE_DIR
from .database import Base, engine
from .routers import detections, export, files
from .services.image_quality import ImageQualityAnalyzer
from .services.ocr_service import OCRService
from .services.upscale import UpscaleEngine
from .websocket_manager import WebSocketManager

Base.metadata.create_all(bind=engine)


def create_app() -> FastAPI:
    app = FastAPI(title="Exploded View OCR", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(files.router)
    app.include_router(detections.router)
    app.include_router(export.router)

    static_dir = BASE_DIR / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    app.state.ocr_service = OCRService()
    app.state.image_quality = ImageQualityAnalyzer()
    app.state.upscale_engine = UpscaleEngine()
    app.state.websocket_manager = WebSocketManager()

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/static/index.html")

    @app.websocket("/ws/updates")
    async def websocket_updates(websocket: WebSocket) -> None:
        manager: WebSocketManager = app.state.websocket_manager
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await manager.disconnect(websocket)

    return app


app = create_app()
