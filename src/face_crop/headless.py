"""Detecção de ambiente sem display e wrapper seguro de imshow."""
import logging

import cv2

logger = logging.getLogger(__name__)


def try_imshow(window: str, frame) -> bool:
    """Tenta mostrar frame em janela. Retorna False se indisponível.

    Em ambientes TCC (sem display), cv2.imshow pode lançar cv2.error.
    O resultado é memoizado pela primeira falha, portanto o custo é pago
    apenas uma vez por worker.
    """
    try:
        cv2.imshow(window, frame)
        cv2.waitKey(1)
        return True
    except cv2.error as e:
        logger.debug("imshow indisponível (%s); entrando em modo headless", e)
        return False