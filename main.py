"""Orquestrador: 1 thread por fonte RTSP."""
import logging
import os
import signal
import sys
import threading

# ATENÇÃO: import utils ANTES de qualquer coisa que puxe cv2
import utils

from src.face_crop.logging_setup import setup_logging
from src.face_crop.worker import Worker

# ================= DEFAULTS =================
OUTPUT_ROOT = "./results"
DEFAULT_SCORE_THRESHOLD = 0.9
DEFAULT_MIN_FACE_SIZE = 100
DEFAULT_DEBUG_SNAPSHOT = 0.0
SAVE_INTERVAL = 1.0
SHOW_FRAMES = True
TRY_GPU = True

# Timeout (segundos) para workers encerrarem antes de forçar saída
WORKER_JOIN_TIMEOUT = 3.0

SOURCES: list[dict] = [
    {
        "url": "rtsp://admin:admin@192.168.0.1/something",
        "name": "cam-name",
    },
]


def _resolve_source_config(src: dict) -> tuple[float, int]:
    logger = logging.getLogger(__name__)
    name = src.get("name", "?")

    score = src.get("score_threshold", DEFAULT_SCORE_THRESHOLD)
    min_size = src.get("min_face_size", DEFAULT_MIN_FACE_SIZE)

    try:
        score = float(score)
        if not (0.0 < score <= 1.0):
            raise ValueError(f"score_threshold fora de (0,1]: {score}")
    except (TypeError, ValueError) as e:
        logger.warning("[%s] score_threshold inválido (%s); usando %s",
                       name, e, DEFAULT_SCORE_THRESHOLD)
        score = DEFAULT_SCORE_THRESHOLD

    try:
        min_size = int(min_size)
        if min_size < 1:
            raise ValueError(f"min_face_size <= 0: {min_size}")
    except (TypeError, ValueError) as e:
        logger.warning("[%s] min_face_size inválido (%s); usando %s",
                       name, e, DEFAULT_MIN_FACE_SIZE)
        min_size = DEFAULT_MIN_FACE_SIZE

    return score, min_size


def main() -> int:
    setup_logging("./logs", logging.INFO)
    logger = logging.getLogger(__name__)

    model_path = utils.setup(output_root=OUTPUT_ROOT)
    logger.info("Modelo: %s", model_path)
    logger.info("SHOW_FRAMES=%s TRY_GPU=%s", SHOW_FRAMES, TRY_GPU)

    stop_event = threading.Event()

    # ----------------------------------------------------------------
    # Handler de sinal: APENAS seta o evento.
    # NUNCA chamar logger/print aqui (deadlock risk em handler do SO).
    # ----------------------------------------------------------------
    _received_signal: list[int] = []

    def _shutdown(signum, _frame):
        _received_signal.append(signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)
    if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break
        signal.signal(signal.SIGBREAK, _shutdown)

    workers: list[Worker] = []
    for src in SOURCES:
        score, min_size = _resolve_source_config(src)
        snapshot = float(src.get("debug_snapshot_seconds", DEFAULT_DEBUG_SNAPSHOT))
        logger.info(
            "[%s] config: score_threshold=%.2f min_face_size=%d debug_snapshot=%.1fs",
            src.get("name", "?"), score, min_size, snapshot,
        )
        workers.append(
            Worker(
                url=src["url"],
                name=src["name"],
                model_path=model_path,
                output_root=OUTPUT_ROOT,
                stop_event=stop_event,
                show=SHOW_FRAMES,
                score_threshold=score,
                save_interval=SAVE_INTERVAL,
                min_face_size=min_size,
                try_gpu=TRY_GPU,
                debug_snapshot_seconds=snapshot,
            )
        )

    logger.info("Iniciando %d worker(s)", len(workers))
    for w in workers:
        w.start()

    # ----------------------------------------------------------------
    # Loop principal: wait com timeout curto para ser responsivo.
    # Ctrl+C NÃO é KeyboardInterrupt aqui (handler customizado), mas
    # o KeyboardInterrupt do default também é capturado por segurança.
    # ----------------------------------------------------------------
    try:
        while not stop_event.is_set():
            # wait curto: loop retoma a cada 0.5s, mesmo se o
            # handler de sinal não conseguir acordar o Event.
            stop_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        # Fallback caso o signal handler não tenha rodado (ex: algum
        # ambiente onde SIGINT não chega ao main thread)
        logger.info("KeyboardInterrupt capturado (fallback)")
        stop_event.set()

    if _received_signal:
        logger.info("Sinal %d recebido; encerrando workers...", _received_signal[0])
    else:
        logger.info("Encerrando workers...")

    # ----------------------------------------------------------------
    # Join com timeout curto. Se algum worker travar (ex: cap.read()
    # bloqueado em RTSP), não esperamos eternamente.
    # ----------------------------------------------------------------
    for w in workers:
        w.join(timeout=WORKER_JOIN_TIMEOUT)

    still_alive = [w.name for w in workers if w.is_alive()]
    if still_alive:
        logger.warning(
            "Workers ainda vivos após %.1fs: %s — forçando saída",
            WORKER_JOIN_TIMEOUT, ", ".join(still_alive),
        )
        # Daemon threads: os._exit mata tudo imediatamente, sem cleanup
        # de atexit. Aceitável aqui porque a saída é intencional.
        os._exit(1)

    logger.info("Encerrado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())