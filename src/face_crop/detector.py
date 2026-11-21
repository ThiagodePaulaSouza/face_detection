"""Wrapper do YuNet via cv2.FaceDetectorYN com seleção de backend."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Detector:
    """Detector facial YuNet. NÃO é thread-safe — 1 instância por worker."""

    def __init__(
        self,
        model_path: str,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        try_gpu: bool = True,
    ) -> None:
        self.model_path = model_path
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self.backend: str = "cpu"
        if try_gpu:
            self._try_setup_gpu()
        self._input_size: tuple[int, int] | None = None

    def _try_setup_gpu(self) -> None:
        try:
            if not hasattr(cv2, "cuda"):
                logger.info("cv2.cuda indisponível — backend CPU")
                return
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count <= 0:
                logger.info("Nenhum device CUDA detectado — backend CPU")
                return
            self._detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            self.backend = "cuda_fp16"
            logger.info("YuNet: backend CUDA FP16 (devices=%d)", count)
        except Exception as e:
            logger.warning("Falha ao configurar CUDA (%s) — backend CPU", e)

    def set_input_size(self, width: int, height: int) -> None:
        """Ajusta input_size. Faz cache para evitar chamadas repetidas."""
        if self._input_size == (width, height):
            return
        self._detector.setInputSize((width, height))
        self._input_size = (width, height)
        logger.debug("YuNet input_size = %dx%d", width, height)

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """Retorna matriz Nx15 [x,y,w,h,l1..l5,score] ou None."""
        h, w = frame.shape[:2]
        self.set_input_size(w, h)
        _, faces = self._detector.detect(frame)
        return faces