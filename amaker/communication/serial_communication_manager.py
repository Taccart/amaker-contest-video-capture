"""
Serial communication manager for aMaker microbot tracker.
Handles all serial port operations in a thread-safe manner.
"""

import logging
import threading
import time

import serial
import serial.tools.list_ports

from amaker.communication.communication_abstract import CommunicationManagerAbstract

# Constants for serial communication
SERIAL_TIMEOUT = 1
SERIAL_DEFAULT_BAUD_RATE = 57600
SERIAL_READ_DELAY = 0.01
THREAD_JOIN_TIMEOUT = 1.0


class SerialCommunicationManagerImpl(CommunicationManagerAbstract):
    """Class to manage serial communication"""

    def __init__(self, serial_port=None, baud_rate=SERIAL_DEFAULT_BAUD_RATE):
        """Initialize the serial manager"""
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial_connection = None
        self.serial_running = False
        self.serial_data = []
        self.serial_thread = None

        if serial_port is not None:
            self.initialize_connection()

    def initialize_connection(self):
        """Initialize serial connection"""
        try:
            if not self.connect():
                logging.warning("Serial connection not established.")
                return False

            self.start_reading()
            return True
        except serial.SerialException as e:
            logging.error(f"Failed to connect to serial port: {e}")
            return False

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
                logging.warning("No serial ports available")
                return False

            logging.info("Available serial ports listed")
            try:
                selection = int(input("\nSelect serial port number (or -1 to skip): "))
                if selection == -1:
                    return False
                port = available_ports[selection]
            except (ValueError, IndexError):
                logging.error("Invalid selection")
                return False

        try:
            self.serial_connection = serial.Serial(port, baud_rate, timeout=SERIAL_TIMEOUT)
            logging.info(f"Connected to {port} at {baud_rate} baud")
            self.serial_port = port
            self.baud_rate = baud_rate
            return True
        except serial.SerialException as e:
            logging.error(f"Failed to connect to serial port: {e}")
            return False

    def start_reading(self):
        """Start the serial reading thread"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_thread = threading.Thread(target=self.read_serial)
            self.serial_thread.daemon = True
            self.serial_thread.start()
            logging.info("Serial thread started")
            return True
        else:
            logging.info("Serial thread NOT started")
            return False

    def read_serial(self):
        """Read data from serial port in a separate thread"""
        self.serial_running = True
        while self.serial_running and self.serial_connection and self.serial_connection.is_open:
            try:
                if self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.readline().decode('utf-8').strip()
                    if data:
                        self.serial_data.append(data)
                        logging.debug(f"Data received: {data}")
                time.sleep(SERIAL_READ_DELAY)  # Small delay to prevent CPU hogging
            except Exception as e:
                logging.error(f"Serial read error: {e}")
                break

    def send_command(self, command):
        """Send a command to the serial port"""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(command.encode())
                self.serial_connection.flush()
                logging.info(f"Sent command: {command.strip()}")
                return True
            except Exception as e:
                logging.error(f"Failed to send command: {e}")
                return False
        else:
            logging.warning(f"Serial connection not available: message not sent was {command}")
            return False

    def get_next_data(self):
        """Get the next data item from the serial buffer"""
        if self.serial_data and len(self.serial_data) > 0:
            return self.serial_data.pop(0)
        return None

    def has_data(self):
        """Check if there is data available"""
        return len(self.serial_data) > 0

    def close(self):
        """Close the serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_running = False
            if self.serial_thread:
                self.serial_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            self.serial_connection.close()
            logging.info("Serial connection closed")
            return True
        return False

    def __del__(self):
        """Destructor to clean up resources"""
        self.close()
