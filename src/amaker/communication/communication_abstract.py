# amaker/communication/manager.py
from abc import ABC, abstractmethod
from typing import Callable


class CommunicationManagerAbstract(ABC):
    """Interface for communication managers"""


    @abstractmethod
    def connect(self, *arg, **kwargs):
        """Connect to a communication channel"""
        pass

    @abstractmethod
    def unregister_on_data_callback(self, callback: Callable):
        """Unregister a callback for data reception"""
        pass
    @abstractmethod
    def register_on_data_callback(self, callback: Callable[[str],None]):
        """Register a callback for data reception"""
        pass

    @abstractmethod
    def send(self, message):
        """Send a command to the communication channel"""
        pass

    @abstractmethod
    def get_next_data(self):
        """Get the next data item from the buffer"""
        pass

    @abstractmethod
    def has_data(self)->bool :
        """Check if there is data available"""
        pass

    @abstractmethod
    def close(self):
        """Close the communication channel"""
        pass

    def __del__(self):
        self.connect()
