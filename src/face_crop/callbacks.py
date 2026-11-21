"""Callback de salvamento de crops com estrutura por fonte e dedup temporal."""
import logging
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    """Normaliza nome para uso seguro como diretório."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
    return cleaned.lower() or "unknown"


class FaceSaver:
    """Salva crops em `{output_root}/{source}/crops/{ts}_score{score}.jpg`.

    Mantém estado por-fonte: 1 crop por `save_interval` segundos.
    """

    def __init__(
        self,
        output_root: str,
        source_name: str,
        *,
        save_interval: float = 1.0,
        min_face_size: int = 40,
    ) -> None:
        self.source_name = source_name
        self.save_interval = save_interval
        self.min_face_size = min_face_size
        self.crop_dir = os.path.join(
            output_root, sanitize_name(source_name), "crops"
        )
        os.makedirs(self.crop_dir, exist_ok=True)

        self._last_save: float = 0.0
        self.saved_count: int = 0

    def __call__(self, frame: np.ndarray, faces: np.ndarray | None) -> None:
        if faces is None or len(faces) == 0:
            return

        now = time.time()
        if now - self._last_save < self.save_interval:
            return

        h, w = frame.shape[:2]

        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            score = float(face[14])

            if fw < self.min_face_size or fh < self.min_face_size:
                continue

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"{ts}_score{score:.2f}.jpg"
            path = os.path.join(self.crop_dir, fname)

            if cv2.imwrite(path, crop):
                self.saved_count += 1
                self._last_save = now
                logger.info(
                    "[%s] Crop salvo: %s (score=%.2f, %dx%d)",
                    self.source_name, fname, score, fw, fh,
                )
                return  # 1 crop por ciclo