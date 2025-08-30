from abc import ABC, abstractmethod
from typing import List

from pyapriltags import Detection


class DetectorAbstract(ABC):
    """
    Interface for detectors : implementations are used to detect objects in a frame.
    """
    @abstractmethod
    def detect(self, **kwargs) -> List[Detection]:
        """Initialize connection"""
        pass
