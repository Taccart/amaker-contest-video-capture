import datetime
import logging
import os
from enum import Enum
from typing import List

import cv2
import numpy as np

# Import SerialManager from the new file
from amaker.serial_manager import SerialManager

# ===== Global Constants =====
# Environment settings
os.environ["QT_QPA_PLATFORM"] = "xcb"

# OpenCV constants
CV_THREADS = 7
WINDOW_TITLE = "aMaker microbot tracker"
DEFAULT_SCREEN_WIDTH = 1280
DEFAULT_SCREEN_HEIGHT = 960
CAMERA_SEARCH_LIMIT = 10

# Video recording constants
VIDEO_CODEC = 'XVID'
VIDEO_FPS = 30.0
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 960

# UI constants
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_BLACK = (0, 0, 0)
UI_VIDEO_TITLE = "Mission Unleash The Bricks"
UI_COLOR_PRIMARY = (250, 250, 250)  # Button fill color
UI_COLOR_SECONDARY = (100, 100, 100)  # Button border color
UI_TEXT_SCALE = 1.2
UI_TEXT_THICKNESS = 2
BUTTON_HEIGHT = 50
BUTTON_WIDTH = 140
BUTTON_X_POS = 1400
BUTTON_MARGIN_X = 5
BUTTON_MARGIN_Y = 10
BUTTON_PADDING = 10
BUTTON_SPACING = 40

# Key codes
KEY_ESC = 27
KEY_CTRL_D = 4
KEY_Q = ord('q')
KEY_F = ord('f')


class BotStatus(Enum):
    """Enum for bot states"""
    UNKNOWN = -1
    WAITING = 0
    MOVING = 1
    SEARCHING = 2
    FETCHING = 3
    CATCHING = 4
    DROPING = 5
    STOPPED = 6
    TO_SAFETY = 10
    MISSON_COMPLETED = 20


# Bot tracker constants
DEFAULT_BOT_COLOR_A = (0, 0, 250)
DEFAULT_BOT_COLOR_B = (0, 0, 255)
DEFAULT_TRAIL_COLOR = (0, 0, 250)
DEFAULT_TRAIL_LENGTH = 20
COMMAND_START = "START"
COMMAND_STOP = "STOP"
COMMAND_SAFETY = "SAFETY"


class LogDisplay:
    def __init__(self, max_logs=5):
        self.logs = []
        self.max_logs = max_logs

    def add_log(self, message):
        self.logs.append(message)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def draw_logs(self, frame):
        for i, log in enumerate(self.logs):
            y_pos = frame.shape[0] - 30 - (i * 20)
            cv2.putText(frame, log, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


class Bot:
    """Class to track a bot's position and color (for identification and video feedback)"""

    def __init__(self, name: str = "microbot", id: int = None, color_a=DEFAULT_BOT_COLOR_A, color_b=DEFAULT_BOT_COLOR_B,
                 trail_color=DEFAULT_TRAIL_COLOR, trail_length: int = DEFAULT_TRAIL_LENGTH):
        self.name = name
        self.id = id
        self.color_a = color_a
        self.color_b = color_b
        self.trail_color = trail_color
        self.trail_length = trail_length
        self.trail = []
        self.status: BotStatus = BotStatus.UNKNOWN
        self.total_distance = 0

    def add_position(self, position):
        """Add a new position to the bot's trail"""
        self.trail.append(position)
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)
        self.total_distance += self.calculate_distance(position)
        logging.debug(f"bot {self.name}:{self.id}, position: {self.get_last_position()}, total distance: {self.total_distance:.2f}")

    def get_last_position(self) -> tuple:
        """Get the last known position of the bot"""
        if self.trail:
            return self.trail[-1]
        else:
            return None

    def set_bot_status(self, state: BotStatus):
        """Set the bot's state"""
        self.status = state
        logging.info(f"Bot {self.name}:{self.id} state changed to {self.status.name}")

    def get_bot_status(self) -> BotStatus:
        """Get the current state of the bot"""
        return self.status

    def calculate_distance(self, position) -> float:
        """Calculate the distance from the last position to the current position"""
        if len(self.trail) < 2:
            return 0
        last_position = self.trail[-2]
        distance = np.linalg.norm(np.array(position) - np.array(last_position))
        return distance

    def get_total_distance(self) -> float:
        """Get the total distance traveled by the bot"""
        return self.total_distance

    def get_trail(self) -> List:
        """Get the bot's trail"""
        return self.trail

    def get_bot_info(self) -> str:
        """Get bot information"""
        return f"{self.name}.{self.id}  is {self.get_bot_status()}"

    def __repr__(self):
        return f"BotTracker(name={self.name}, color_a={self.color_a}, color_b={self.color_b}, trail_color={self.trail_color}, trail_length={self.trail_length})"


class AmakerBotTracker():
    def __init__(self, camera_index: int = 0, bot_trackers: List[Bot] = None, serial_manager=None):
        self.logHistory = LogDisplay()
        self.logHistory.add_log("Initializing...")
        self.serial_manager = serial_manager
        if serial_manager is not None:
            logging.info("Serial communication activated.")
        else:
            logging.info("Serial communication not activated.")

        if not isinstance(bot_trackers, list):
            raise ValueError("No bot trackers provided.")
        if len(bot_trackers) < 1:
            raise ValueError("No bot trackers provided.")
        self.tracked_bots = bot_trackers
        self.camera_index = self.camera_choice() if camera_index <= 0 else camera_index

        self.video_capture = cv2.VideoCapture(self.camera_index)
        cv2.setLogLevel(2)  # Set OpenCV log severity to no logs
        if not self.video_capture.isOpened():
            raise ValueError(f"Camera {camera_index} not found or cannot be opened.")

    def camera_choice(self) -> int:
        """Select a camera from available cameras"""
        index = 0
        available_cameras = []
        logging.info(f"Searching for cameras...")
        while True:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    logging.debug(f"Camera {index} : not ok.")
                else:
                    logging.info(f"Camera {index} : ok.")
                    available_cameras.append(index)
                    cap.release()
            except Exception as e:
                logging.warning(f"Camera {index}: error {e}")
            if index > CAMERA_SEARCH_LIMIT:
                break
            index += 1

        if not available_cameras:
            logging.error("No cameras found!")
            exit()

        # choose camera to be used 
        print("\nAvailable cameras:")
        for cam in available_cameras:
            logging.info(f"Camera {cam}")

        selected_camera = int(input("\nSelect the camera index to use: "))
        if selected_camera not in available_cameras:
            logging.error("Invalid camera index selected!")
            exit()
        return selected_camera

    # Button callback functions
    def button_start(self):
        """Handle start button click"""

        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_START)
            self.logHistory.add_log(f"sent: {COMMAND_START}")
        else:
            self.logHistory.add_log(f"unsent: {COMMAND_START}")

    def button_stop(self):
        """Handle stop button click"""
        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_STOP)
            self.logHistory.add_log(f"sent: {COMMAND_STOP}")
        else:
            self.logHistory.add_log(f"unsent: {COMMAND_STOP}")

    def button_safety(self):
        """Handle safety button click"""
        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_SAFETY)
            self.logHistory.add_log(f"sent: {COMMAND_SAFETY}")
        else:
            self.logHistory.add_log(f"unsent: {COMMAND_SAFETY}")

    def process_bot_tracking(self, frame):
        for bot in self.tracked_bots:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bot_mask = cv2.inRange(rgb, bot.color_a, bot.color_b)
                bot_contours, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(bot_contours) > 0:
                    self._update_bot_position(bot, bot_contours, frame)

                else:
                    logging.debug(f"No contours found for {bot.name}")
            except Exception as e:
                logging.error(f"Error tracking {bot.name}: {e}")

    def _add_button(self, button, frame):
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_COLOR_PRIMARY, -1)  # Filled rectangle
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_COLOR_SECONDARY, 2)  # Border
        cv2.putText(frame, button['text'], (button['x'] + BUTTON_PADDING, button['y'] + 37), cv2.FONT_HERSHEY_SIMPLEX,
                    UI_TEXT_SCALE, TEXT_COLOR_BLACK, UI_TEXT_THICKNESS)

    def _build_display_frame(self, input_frame, buttons=None):
        """Build the display frame with UI elements"""
        display_frame = input_frame.copy()  # Make a copy for display
        if self.is_fullscreen:
            display_frame = cv2.resize(input_frame, (self.screen_width, self.screen_height))
        else:
            # Use current window dimensions or default to original frame size
            display_frame = cv2.resize(input_frame, (self.window_width, self.window_height))

        # Display serial data if available - using SerialManager
        if self.serial_manager and self.serial_manager.has_data():
            serial_data = self.serial_manager.get_next_data()
            if serial_data:
                cv2.putText(input_frame, f"Serial: {serial_data}", (10, input_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, UI_TEXT_SCALE, (0, 255, 0), 1)
                logging.info(serial_data)

        cv2.putText(display_frame, UI_VIDEO_TITLE, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, TEXT_COLOR_WHITE, 5,
                    cv2.LINE_AA)

        # Draw buttons
        for button in buttons:
            self._add_button(button, display_frame)

        return display_frame

    def _setup_video_recording(self, recording):
        video_writer = None
        if recording:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f'aMaker_microbot_tracker_{current_time}.avi'
            video_writer = cv2.VideoWriter(video_name, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
            logging.info(f"Video recording started: {video_name}")
        else:
            logging.info("No video recording.")
        return video_writer

    def _cleanup_resources(self, video_writer):
        if self.serial_manager:
            self.serial_manager.close()
            logging.info("Serial port closed.")
        self.video_capture.release()
        logging.info("Video capture released.")
        if video_writer:
            # Release the video writer
            video_writer.release()
            logging.info("Video saved successfully.")

    def _initialize_window(self, window_name):
        ret, input_frame = self.video_capture.read()
        if not ret:
            raise Exception("Failed to get video capture.")

        original_height, original_width = input_frame.shape[:2]

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_OPENGL, 0)  # Disable OpenGL status bar
        cv2.resizeWindow(window_name, original_width, original_height)  # Start with original frame size
        cv2.setNumThreads(CV_THREADS)  # Set the number of threads for OpenCV

        return (input_frame, original_height, original_width)

    def _create_buttons(self, window_name):
        # Define buttons
        button_start = {'x': BUTTON_X_POS + BUTTON_MARGIN_X, 'y': BUTTON_MARGIN_Y, 'w': BUTTON_WIDTH,
                        'h': BUTTON_HEIGHT, 'text': 'Start', 'clicked': False}
        button_stop = {'x': BUTTON_X_POS + BUTTON_MARGIN_X + BUTTON_WIDTH + BUTTON_SPACING, 'y': BUTTON_MARGIN_Y,
                       'w': BUTTON_WIDTH, 'h': BUTTON_HEIGHT, 'text': 'Stop', 'clicked': False}
        button_safety = {'x': BUTTON_X_POS + BUTTON_MARGIN_X + 2 * (BUTTON_WIDTH + BUTTON_SPACING),
                         'y': BUTTON_MARGIN_Y, 'w': BUTTON_WIDTH, 'h': BUTTON_HEIGHT, 'text': 'Safety',
                         'clicked': False}

        def mouse_callback(event, x, y, flags, param):
            """Handle mouse events"""
            if event == cv2.EVENT_LBUTTONDOWN:
                if (button_safety['x'] <= x <= button_safety['x'] + button_safety['w'] and button_safety['y'] <= y <=
                        button_safety['y'] + button_safety['h']):
                    self.button_safety()  # Custom method to handle button click
                elif (button_start['x'] <= x <= button_start['x'] + button_start['w'] and button_start['y'] <= y <=
                      button_start['y'] + button_start['h']):
                    self.button_start()
                elif (button_stop['x'] <= x <= button_stop['x'] + button_stop['w'] and button_stop['y'] <= y <=
                      button_stop['y'] + button_stop['h']):
                    self.button_stop()

        # Set the mouse callback
        cv2.setMouseCallback(window_name, mouse_callback)
        return [button_start, button_stop, button_safety]

    def _main_tracking_loop(self, window_name, buttons, video_writer):
        while True:
            ret, input_frame = self.video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break

            # Convert to HSV for color detection
            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGRA2RGB)

            for bot_tracker in self.tracked_bots:
                # Get the bot's color range
                bot_mask = cv2.inRange(rgb, bot_tracker.color_a, bot_tracker.color_b)
                bot_contours, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Extract bot positions
                if bot_contours:
                    x, y, w, h = cv2.boundingRect(bot_contours[0])
                    bot_position = (x + w // 2, y + h // 2)
                    bot_tracker.add_position(bot_position)

                    if len(bot_tracker.get_trail()) > 1:
                        cv2.polylines(input_frame, [np.array(bot_tracker.get_trail())], False, bot_tracker.trail_color,
                                      2)
                        cv2.circle(input_frame, bot_position, 5, bot_tracker.trail_color, -1)

                        cv2.putText(input_frame, f"{bot_tracker.name}", bot_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    bot_tracker.trail_color, 1, cv2.LINE_4)

            # Add text to the frame

            if video_writer:
                # Save the frame to the video file
                video_writer.write(input_frame)

            # Show the video feed
            display_frame = self._build_display_frame(input_frame, buttons)

            for i, bot_tracker in enumerate(self.tracked_bots):
                y_pos = 30 + (i * 25)  # Stagger vertical positions
                cv2.putText(display_frame, bot_tracker.get_bot_info(), (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            bot_tracker.trail_color, 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_Q or key == KEY_CTRL_D or key == KEY_ESC:  # 'q' or Ctrl+D or ESC to quit
                break

    def start_tracking_new(self, recording=False, window_name=WINDOW_TITLE):
        """Start tracking bots in video feed"""
        self._initialize_window(window_name)
        buttons = self._create_buttons(window_name)
        video_writer = self._setup_video_recording(recording)

        try:
            self._main_tracking_loop(window_name, buttons, video_writer)
        finally:
            self._cleanup_resources(video_writer)

    def start_tracking(self, recording: bool = False, window_name=WINDOW_TITLE):
        """Start tracking bots in video feed"""
        input_frame, original_height, original_width = self._initialize_window(window_name)
        buttons = self._create_buttons(window_name)

        self.is_fullscreen = True
        original_height, original_width = input_frame.shape[:2]
        self.window_width, self.window_height = original_width, original_height  # Initialize with frame size

        try:
            # Try to get screen info - this is optional and only works if screeninfo is installed
            from screeninfo import get_monitors
            monitor = get_monitors()[0]
            self.screen_width, self.screen_height = monitor.width, monitor.height
            logging.info(f"Detected screen size: {self.screen_width}x{self.screen_height}")
        except:
            # Fallback to reasonable defaults if we can't get screen info
            self.screen_width, self.screen_height = DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT
            logging.warning(f"Couldn't detect screen size. Using defaults: {self.screen_width}x{self.screen_height}")

        video_writer = self._setup_video_recording(recording)

        logging.info(f"Bot trackers: {self.tracked_bots}")
        logging.info(f"Press 'q' or ESC to quit.")
        logging.info(f"Press 'f' to toggle full screen.")

        self._main_tracking_loop(window_name, buttons, video_writer)

        self._cleanup_resources(video_writer)

    def __del__(self):
        """Destructor cleans up resources"""
        try:
            if self.serial_manager:
                self.serial_manager.close()
        except Exception as e:
            logging.error(f"Error during serial cleanup: {e}")
        try:
            if hasattr(self, 'video_capture') and self.video_capture.isOpened():
                self.video_capture.release()
                logging.info("Video capture released")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        try:
            cv2.destroyAllWindows()
            logging.info("All windows destroyed")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    bot1 = Bot(name="redBot", color_a=(106, 40, 30), color_b=(150, 70, 60), trail_color=(0,0,255))
    bot1 = Bot(name="blueBot", color_a=(106, 40, 30), color_b=(150, 70, 60), trail_color=(255,128,0))
    bot2 = Bot(name="yellowBot", color_a=(150, 106, 30), color_b=(150, 150, 60), trail_color=(0,255,255))

    # Create serial manager
    serial_manager = SerialManager()
    serial_manager.connect_serial(baud_rate=57600, port="/dev/ttyACM0")

    # Create video tracker with serial manager
    vt = AmakerBotTracker(camera_index=4, bot_trackers=[bot1, bot2], serial_manager=serial_manager)
    vt.start_tracking(recording=True)
