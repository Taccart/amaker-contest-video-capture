# amaker/communication/manager.py
from abc import ABC, abstractmethod


class CommunicationManagerAbstract(ABC):
    """Interface for communication managers"""

    @abstractmethod
    def initialize_connection(self):
        """Initialize connection"""
        pass

    @abstractmethod
    def list_available_channels(self):
        """List all available communication channels"""
        pass

    @abstractmethod
    def connect(self, *arg, **kwargs):
        """Connect to a communication channel"""
        pass

    @abstractmethod
    def start_reading(self):
        """Start reading from the communication channel"""
        pass

    @abstractmethod
    def send_command(self, command):
        """Send a command to the communication channel"""
        pass

    @abstractmethod
    def get_next_data(self):
        """Get the next data item from the buffer"""
        pass

    @abstractmethod
    def has_data(self):
        """Check if there is data available"""
        pass

    @abstractmethod
    def close(self):
        """Close the communication channel"""
        pass

    def __del__(self):
        self.connect()
