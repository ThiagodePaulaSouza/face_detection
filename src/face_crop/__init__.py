"""Pipeline de detecção e crop facial com YuNet."""
from .callbacks import FaceSaver, sanitize_name
from .detector import Detector
from .feed import Feed
from .worker import Worker

__all__ = ["Detector", "Feed", "Worker", "FaceSaver", "sanitize_name"]