"""Wrapper de VideoCapture RTSP com reconexão automática."""
import logging
import re
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _mask_url(url: str) -> str:
    return re.sub(r"(rtsp://[^:/@]+:)[^@]+(@)", r"\1***\2", url)


class Feed:
    """Leitor RTSP que reconecta automaticamente em caso de falha."""

    # Timeout (ms) para abrir o stream e para ler cada frame
    OPEN_TIMEOUT_MS = 5000
    READ_TIMEOUT_MS = 5000

    def __init__(
        self,
        url: str,
        name: str,
        *,
        reconnect_delay: float = 2.0,
    ) -> None:
        self.url = url
        self.name = name
        self.reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> bool:
        logger.info("[%s] Abrindo RTSP: %s", self.name, _mask_url(self.url))
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.READ_TIMEOUT_MS,
        ]
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        # Buffer mínimo reduz latência (quando suportado)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.warning("[%s] Falha ao abrir RTSP", self.name)
            cap.release()
            return False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("[%s] Conectado: %dx%d", self.name, w, h)
        self._cap = cap
        return True

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            time.sleep(self.reconnect_delay)
            if not self._open():
                return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("[%s] Leitura falhou; reconectando", self.name)
            self._release()
            return None
        return frame

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def close(self) -> None:
        self._release()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()