from PyQt6.QtCore import QObject, pyqtSignal, QThread

class CommunicationSignalEmitter(QObject):
    """Thread-safe wrapper that emits Qt signals when data arrives"""
    data_received = pyqtSignal(str)  # Emits received data

    def __init__(self, communication_manager):
        super().__init__()
        self.communication_manager = communication_manager
        self.running = False

    def start_listening(self):
        """Start listening thread"""
        self.running = True
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._listen_loop)
        self.thread.start()

    def _listen_loop(self):
        """Run in background thread, emit signals"""
        while self.running and self.communication_manager:
            if self.communication_manager.has_data():
                data = self.communication_manager.get_next_data()
                if data:
                    self.data_received.emit(data)  # Thread-safe signal
            QThread.msleep(10)  # Small delay

    def stop_listening(self):
        """Stop listening thread"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.quit()
            self.thread.wait()
