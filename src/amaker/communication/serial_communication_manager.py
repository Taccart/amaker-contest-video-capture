"""
Serial communication manager for aMaker microbot tracker.
Handles all serial port operations in a feed_thread-safe manner.
"""

import logging
import threading
import time
from queue import Queue
import queue
from typing import Callable

import serial
import serial.tools.list_ports

from amaker.communication.communication_abstract import CommunicationManagerAbstract

# Constants for serial communication
SERIAL_TIMEOUT = 1
SERIAL_DEFAULT_BAUD_RATE = 57600
SERIAL_READ_DELAY = 0.1
THREAD_JOIN_TIMEOUT = 1.0
DEFAULT_MAX_QUEUE_SIZE=100

class SerialCommunicationManagerImpl(CommunicationManagerAbstract):
    """Implementation classs for a  the serial communication manager.
    Will be used with a microcontroller connected to USB.
    r"""

    def __init__(self, serial_port=None, baud_rate=SERIAL_DEFAULT_BAUD_RATE, max_queue_size=DEFAULT_MAX_QUEUE_SIZE):
        """Initialize the serial manager"""
        self._LOG = logging.getLogger(__name__)
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        self.serial = None


        self.serial_connection = None
        self.serial_data = Queue(maxsize=max_queue_size)
        self.serial_thread = None
        self.on_data_received: Callable[[str],None]|None = None

        self.callbacks_lock = threading.Lock()
        self.connect()  # Add explicit connection
        time.sleep(0.2)




    def list_available_channels(self):
        """List all available serial ports"""
        ports = serial.tools.list_ports.comports()
        available_ports = []

        print("\nAvailable serial ports:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device} - {port.description}")
            available_ports.append(port.device)

        return available_ports

    def connect(self, *arg, **kwargs):
        """Connect to a serial port"""
        port = self.serial_port
        baud_rate = self.baud_rate

        if not port:
            available_ports = self.list_available_channels()
            if not available_ports:
                self._LOG.warning("No serial ports available")
                return False

            self._LOG.info("Available serial ports listed")
            try:
                selection = int(input("\nSelect serial port number (or -1 to skip): "))
                if selection == -1:
                    return False
                port = available_ports[selection]
            except (ValueError, IndexError):
                self._LOG.error("Invalid selection")
                return False

        try:
            self.serial_connection = serial.Serial(port, baud_rate, timeout=SERIAL_TIMEOUT)
            self._LOG.info(f"Connected to {port} at {baud_rate} baud")
            self.serial_port = port
            self.baud_rate = baud_rate
            if not self.serial_connection.is_open:
                self.serial_connection.open()
            return True
        except serial.SerialException as e:
            self._LOG.error(f"Failed to connect to serial port: {e}")
            return False


    def start_reading(self):
        """Start the serial reading feed_thread"""
        self._LOG.info ("start reading")
        if not self.serial_connection or not self.serial_connection.is_open:
            self._LOG.error("Cannot start reading: serial connection not open")
            return False
        self.serial_thread = threading.Thread(target=self.read_serial)
        self.serial_thread.name = "SerialReaderThread"
        self.serial_thread.daemon = True
        self.serial_thread.start()
        wait_start = time.time()
        max_wait = 5.0  # Maximum 5 seconds to wait

        while not self.serial_thread.is_alive() and time.time() - wait_start < max_wait:
            self._LOG.info(f"Waiting for feed_thread to start...")
            time.sleep(1)
        if not self.serial_thread.is_alive():
            self._LOG.error("Serial feed_thread failed to start within timeout")
            return False
        self._LOG.info(f"Serial feed_thread started successfully ? {self.serial_thread.is_alive()}")
        return True

    def read_serial(self):
        """Read data from serial port in a separate feed_thread"""
        self._LOG.info("read serial")
        while self.serial_connection and self.serial_connection.is_open:
            try:
                if self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.readline().decode('utf-8').strip()
                    if data:
                        try:
                            # Add to queue without blocking (fail if full)
                            self.serial_data.put_nowait(data)
                        except queue.Full:  # Use queue.Full, not Queue.Full
                            self._LOG.error(f"Serial queue full, discarding data: {data}")
                        if self.on_data_received:
                            self.on_data_received(data)  # Pass data directly, not get_next_data()
                time.sleep(SERIAL_READ_DELAY)
            except Exception as e:
                self._LOG.error(f"Serial read error: {str(e)}")
                break

    def send(self, message):
        """Send a command to the serial port"""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(message.encode())
                self.serial_connection.flush()
                self._LOG.info(f"Sent command: {message.strip()}")
                return True
            except Exception as e:
                self._LOG.error(f"Failed to send command: {e}")
                return False
        else:
            self._LOG.warning(f"Serial connection not available: message not sent was {message}")
            return False

    def get_next_data(self):
        """Get the next data item from the serial buffer"""
        try:
            # Non-blocking get
            return self.serial_data.get_nowait()
        except queue.Empty:
            return None

    def has_data(self):
        """Check if there is data available"""
        return self.serial_data.not_empty

    def register_on_data_callback(self, callback):
        """Register a function to be called when data is received
        The callback function must accept a single string parameter
        """

        if self.on_data_received:
            self._LOG.warning("Callback already registered, ignoring new one")
        if callable(callback):
            self._LOG.debug(f"Registering callback for data received on {callback}")
            with self.callbacks_lock:
                self.on_data_received=callback
                return True
        self._LOG.warning("Callback is not callable: ignoring demand.")
        return False

    def unregister_on_data_callback(self, callback):
        """Remove a previously registered callback function"""
        if self.on_data_received:
            self.on_data_received = None
            self._LOG.info("Callback unregistered.")
        else:
            self._LOG.info("No callback to unregister.")

    def close(self):
        """Close the serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            if self.serial_thread:
                self.serial_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            self.serial_connection.close()
            self._LOG.info("Serial connection closed")
            return True
        return False

    def __del__(self):
        """Destructor to clean up resources"""
        self.close()
