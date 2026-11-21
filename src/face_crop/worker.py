"""Thread por câmera: feed + detector + saver."""
import logging
import os
import threading
import time

import cv2
import numpy as np

from .callbacks import FaceSaver
from .detector import Detector
from .feed import Feed
from .headless import try_imshow

logger = logging.getLogger(__name__)


class Worker(threading.Thread):
    """Roda 1 câmera até stop_event ser setado.

    Cada instância cria seu próprio Detector, FaceSaver e Feed — não
    compartilha estado com outros workers (thread-safe por isolamento).
    """

    # Intervalo (segundos) entre logs de progresso "frames lidos"
    PROGRESS_LOG_INTERVAL = 30.0

    def __init__(
        self,
        *,
        url: str,
        name: str,
        model_path: str,
        output_root: str,
        stop_event: threading.Event,
        show: bool = False,
        score_threshold: float = 0.6,
        save_interval: float = 1.0,
        min_face_size: int = 40,
        try_gpu: bool = True,
        debug_snapshot_seconds: float = 0.0,
    ) -> None:
        super().__init__(name=f"worker-{name}", daemon=True)
        self.url = url
        self.source_name = name
        self.model_path = model_path
        self.output_root = output_root
        self.stop_event = stop_event
        self.show = show
        self.score_threshold = score_threshold
        self.save_interval = save_interval
        self.min_face_size = min_face_size
        self.try_gpu = try_gpu
        self.debug_snapshot_seconds = debug_snapshot_seconds

        # Estado interno
        self._imshow_ok = show
        self._last_snapshot: float = 0.0
        self._debug_dir = os.path.join(output_root, self.source_name, "debug")

    # ------------------------------------------------------------------
    # Snapshot de debug (opcional, útil em TCC/headless)
    # ------------------------------------------------------------------
    def _maybe_snapshot(self, frame: np.ndarray, faces: np.ndarray | None) -> None:
        """Salva frame anotado periodicamente em `{output}/{source}/debug/`.

        Desligado quando `debug_snapshot_seconds <= 0`.
        """
        if self.debug_snapshot_seconds <= 0:
            return

        now = time.time()
        if now - self._last_snapshot < self.debug_snapshot_seconds:
            return

        os.makedirs(self._debug_dir, exist_ok=True)
        annotated = frame.copy()

        if faces is not None and len(faces) > 0:
            for f in faces:
                x, y, w, h = f[:4].astype(int)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{float(f[14]):.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._debug_dir, f"snapshot_{ts}.jpg")
        if cv2.imwrite(path, annotated):
            self._last_snapshot = now
            logger.debug("[%s] snapshot: %s", self.source_name, path)

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("[%s] Worker iniciado", self.source_name)

        detector: Detector | None = None
        saver: FaceSaver | None = None
        feed: Feed | None = None
        frames_read = 0
        last_progress_log = time.time()

        try:
            detector = Detector(
                model_path=self.model_path,
                score_threshold=self.score_threshold,
                try_gpu=self.try_gpu,
            )
            saver = FaceSaver(
                output_root=self.output_root,
                source_name=self.source_name,
                save_interval=self.save_interval,
                min_face_size=self.min_face_size,
            )
            feed = Feed(self.url, self.source_name)

            while not self.stop_event.is_set():
                frame = feed.read()
                if frame is None:
                    continue

                frames_read += 1

                # Log de progresso a cada N segundos (confirma stream vivo)
                now = time.time()
                if now - last_progress_log >= self.PROGRESS_LOG_INTERVAL:
                    logger.info(
                        "[%s] frames lidos até agora: %d",
                        self.source_name,
                        frames_read,
                    )
                    last_progress_log = now

                faces = detector.detect(frame)
                saver(frame, faces)
                self._maybe_snapshot(frame, faces)

                if self._imshow_ok:
                    self._imshow_ok = try_imshow(
                        f"RTSP - {self.source_name}", frame
                    )

        except Exception:
            logger.exception("[%s] Erro fatal no worker", self.source_name)

        finally:
            if feed is not None:
                feed.close()

            # Só tenta fechar janela se imshow chegou a funcionar alguma vez
            if self._imshow_ok:
                try:
                    cv2.destroyWindow(f"RTSP - {self.source_name}")
                except cv2.error:
                    pass

            saved = saver.saved_count if saver else 0
            backend = detector.backend if detector else "?"
            logger.info(
                "[%s] Worker encerrado (frames=%d, crops=%d, backend=%s)",
                self.source_name,
                frames_read,
                saved,
                backend,
            )