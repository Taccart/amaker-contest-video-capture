from PyQt6.QtCore import QObject, pyqtSignal, QTimer

class CommunicationSignalEmitter(QObject):
    """Thread-safe wrapper that emits Qt signals when data arrives"""
    data_received = pyqtSignal(str)  # Emits received data

    def __init__(self, communication_manager):
        super().__init__()
        self.communication_manager = communication_manager
        self.running = False
        
        # Use QTimer instead of QThread for polling
        # This keeps everything in the main Qt thread
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._check_data)

    def start_listening(self):
        """Start listening using QTimer (main thread)"""
        self.running = True
        self.poll_timer.start(10)  # Check every 10ms

    def _check_data(self):
        """Check for data in main thread and emit signals"""
        if not self.running or not self.communication_manager:
            return
            
        if self.communication_manager.has_data():
            data = self.communication_manager.get_next_data()
            if data:
                self.data_received.emit(data)  # Safe to emit from main thread

    def stop_listening(self):
        """Stop listening"""
        self.running = False
        if self.poll_timer.isActive():
            self.poll_timer.stop()
