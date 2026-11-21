"""Setup compartilhado: env vars, diretórios, download do modelo.

IMPORTANTE: este módulo deve ser importado ANTES de qualquer módulo
que puxe `cv2`, para que CUDA_VISIBLE_DEVICES tenha efeito.
"""
import logging
import os
import urllib.request
from pathlib import Path

# --- Env vars (DEVEM preceder `import cv2` em qualquer lugar do projeto) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # GPU 0 já tem outro processo
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # silencia FFmpeg

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


def setup(output_root: str = "./results", model_dir: str = ".") -> str:
    """Cria diretórios e garante que o modelo .onnx existe.

    Retorna o caminho absoluto do modelo.
    """
    Path(output_root).mkdir(parents=True, exist_ok=True)
    model_path = os.path.abspath(os.path.join(model_dir, MODEL_FILENAME))

    if not os.path.isfile(model_path):
        logger.info("Modelo não encontrado; baixando de %s", MODEL_URL)
        tmp = model_path + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, model_path)
        logger.info("Modelo salvo: %s", model_path)
    else:
        logger.info("Modelo encontrado: %s", model_path)

    return model_path