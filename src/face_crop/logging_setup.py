"""Configuração de logging estruturado com rotação de arquivo."""
import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_dir: str = "./logs", level: int = logging.INFO) -> None:
    """Configura root logger com console + arquivo rotacionado."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-18s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Limpa handlers prévios (re-execução)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/face_crop.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Silencia libs barulhentas
    logging.getLogger("urllib3").setLevel(logging.WARNING)