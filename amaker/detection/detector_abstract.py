from abc import ABC, abstractmethod
from typing import List

from pyapriltags import Detection


class DetectorAbstract(ABC):
    @abstractmethod
    def detect(self, **kwargs) -> List[Detection]:
        """Initialize connection"""
        pass
