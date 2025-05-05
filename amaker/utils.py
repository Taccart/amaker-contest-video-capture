
class LogDisplay:
    def __init__(self, max_logs=5):
        self.logs = []
        self.max_logs = max_logs

    def add_log(self, message):
        current_time = datetime.datetime.now().strftime("%H%M%S")
        self.logs.append(current_time+" "+str(message))
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def draw_logs(self, frame):
        if self.logs:
            for i, log in enumerate(reversed(self.logs)):
                y_pos = frame.shape[0] - 30 - (i * 20)
                cv2.putText(frame, log, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

