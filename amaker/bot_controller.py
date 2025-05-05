import datetime
import logging
import os
from typing import List

import cv2
import numpy as np

# Import SerialManager from the new file
from amaker.serial_manager import SerialManager
from amaker.bot import Bot

# ===== Global Constants =====
# Environment settings
os.environ["QT_QPA_PLATFORM"] = "xcb"
# exit codes
EXIT_NO_CAMERA=1
EXIT_INVALID_CAMERA_CHOICE=2
# OpenCV constants
CV_THREADS = 4
# Window settings
WINDOW_TITLE = "aMaker microbot tracker"
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080
#CAMERA_SEARCH_LIMIT
CAMERA_SEARCH_LIMIT = 10

# Video recording constants
VIDEO_CODEC = 'XVID'
VIDEO_FPS = 30.0
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# UI constants
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_BLACK = (0, 0, 0)
UI_VIDEO_TITLE = "Mission Unleash The Bricks"
UI_COLOR_PRIMARY = (250, 250, 250)  # Button fill color
UI_COLOR_SECONDARY = (100, 100, 100)  # Button border color
UI_TEXT_SCALE = 1.2
UI_TEXT_THICKNESS = 2
UI_DEFAULT_IMAGE_MASK= '/home/taccart/VSCode/amaker-path-capture/resources/background.png'
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




# Bot tracker constants

COMMAND_START = "123456789-12345679-123456789-123456789-"
COMMAND_STOP = "STOP"
COMMAND_SAFETY = "SAFETY"


class LogDisplay:
    def __init__(self, max_logs=8):
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
                cv2.putText(frame, log, (10, y_pos), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 1)

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
        self.camera_index = self.user_input_camera_choice() if camera_index <= 0 else camera_index

        self.video_capture = cv2.VideoCapture(self.camera_index)
        cv2.setLogLevel(2)  # Set OpenCV log severity to no logs
        if not self.video_capture.isOpened():
            raise ValueError(f"Camera {camera_index} not found or cannot be opened.")


    def bot_assign_colors(self):
        ## broadcast new colors of all bots
        raise NotImplementedError
    def bot_verify_colors(self):
        ## check aknowledgement of new colors for all bots
        raise NotImplementedError
    def _update_bot_position(self):
        raise NotImplementedError

    def user_input_camera_choice(self) -> int | None:
        """Select a camera from available cameras"""
        index = 0
        available_cameras = []
        logging.info(f"Searching for cameras...")
        cap=None
        while True:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    logging.debug(f"Camera {index} : not ok.")
                else:
                    logging.info(f"Camera {index} : ok.")
                    available_cameras.append(index)

            except Exception as e:
                logging.warning(f"Camera {index}: error {e}")
            finally:
                try:
                    if cap:
                        cap.release()
                except Exception as e:
                    logging.warning(f"Camera {index}: error releasing {e}")
            if index > CAMERA_SEARCH_LIMIT:
                break
            index += 1

        if not available_cameras:
            logging.error("No cameras found!")
            exit(EXIT_NO_CAMERA)

        # choose camera to be used 
        print("\nAvailable cameras:")
        for cam in available_cameras:
            logging.info(f"Camera {cam}")

        selected_camera = int(input("\nSelect the camera index to use: "))
        if selected_camera not in available_cameras:
            logging.error("Invalid camera index selected!")
            exit(EXIT_INVALID_CAMERA_CHOICE)
        else:
            return selected_camera

    # Button callback functions
    def _on_button_start(self):
        """Handle start button click"""

        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_START)
            self.logHistory.add_log(f"sent: {COMMAND_START}")
        else:
            self.logHistory.add_log(f"unsent: {COMMAND_START}")

    def _on_button_stop(self):
        """Handle stop button click"""
        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_STOP)
            self.logHistory.add_log(f"sent: {COMMAND_STOP}")
        else:
            self.logHistory.add_log(f"unsent: {COMMAND_STOP}")

    def _on_button_safety(self):
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

    @staticmethod
    def _add_button(button, frame):
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
    # Load transparent PNG image (do this once in __init__ for efficiency)
        if not hasattr(self, 'logo_image'):
            # Load with transparency preserved
            self.logo_image = cv2.imread(UI_DEFAULT_IMAGE_MASK, cv2.IMREAD_UNCHANGED)
            # Optional: resize if needed
            # self.logo_image = cv2.resize(self.logo_image, (width, height))

        # Add the transparent image to display_frame
        if hasattr(self, 'logo_image') and self.logo_image is not None:
            # Position in top-right corner (adjust x,y as needed)
            x = display_frame.shape[1] - self.logo_image.shape[1] - 20
            y = 20
            self._overlay_transparent_image(display_frame, self.logo_image, x, y)

        # Your existing c   ode for buttons, text, etc.

        return display_frame

    def _setup_video_recording(self, recording)-> cv2.VideoWriter | None:
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

    def _overlay_transparent_image(self, background, overlay, x, y):
        """
        Overlay a transparent PNG image onto the background

        Args:
            background: The background frame
            overlay: The transparent PNG image with alpha channel
            x, y: Top-left position to place the overlay
        """
        # Get image sizes
        h, w = overlay.shape[:2]

        # Check if overlay exceeds frame boundaries
        if y + h > background.shape[0] or x + w > background.shape[1]:
            # Crop overlay to fit within frame
            h = min(h, background.shape[0] - y)
            w = min(w, background.shape[1] - x)
            overlay = overlay[:h, :w]

        # Get the alpha channel and RGB channels
        if overlay.shape[2] == 4:  # With alpha channel
            alpha = overlay[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            rgb = overlay[:, :, :3]

            # Calculate the foreground and background
            foreground = rgb * alpha
            background_area = background[y:y+h, x:x+w, :3] * (1 - alpha)

            # Combine foreground and background
            background[y:y+h, x:x+w, :3] = foreground + background_area

    def _cleanup_resources(self, video_writer):
        try:
            if self.serial_manager:
                self.serial_manager.close()
                logging.info("Serial port closed.")
        except Exception as e:
            logging.warning(f"Error during serial cleanup: {e}")

        try:
            self.video_capture.release()
            logging.info("Video capture released.")
        except Exception as e:
            logging.warning(f"Error during video capture release: {e}")

        try:
            if video_writer:
                # Release the video writer
                video_writer.release()
                logging.info("Video writer released successfully.")
        except Exception as e:
            logging.warning(f"Error during video capture release: {e}")


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

    def _create_buttons_list(self, window_name):
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
                    self._on_button_safety()  # Custom method to handle button click
                elif (button_start['x'] <= x <= button_start['x'] + button_start['w'] and button_start['y'] <= y <=
                      button_start['y'] + button_start['h']):
                    self._on_button_start()
                elif (button_stop['x'] <= x <= button_stop['x'] + button_stop['w'] and button_stop['y'] <= y <=
                      button_stop['y'] + button_stop['h']):
                    self._on_button_stop()

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
            self.logHistory.draw_logs(display_frame)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_Q or key == KEY_CTRL_D or key == KEY_ESC:  # 'q' or Ctrl+D or ESC to quit
                break


    def start_tracking(self, recording: bool = False, window_name=WINDOW_TITLE):
        """Start tracking bots in video feed"""
        input_frame, original_height, original_width = self._initialize_window(window_name)
        buttons = self._create_buttons_list(window_name)

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
        video_writer=None
        try:
            video_writer = self._setup_video_recording(recording)
            logging.info(f"Bot trackers: {self.tracked_bots}")
            logging.info(f"Press 'q' or ESC to quit.")
            logging.info(f"Press 'f' to toggle full screen.")
            self._main_tracking_loop(window_name, buttons, video_writer)
        except Exception as e:
            logging.error(f"Error during video tracking: {e}")
        finally:

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
            logging.error(f"Error during capture cleanup: {e}")
        try:
            cv2.destroyAllWindows()
            logging.info("All windows destroyed")
        except Exception as e:
            logging.error(f"Error during winwow cleanup: {e}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    bot1 = Bot(name="redBot", color_a=(106, 40, 30), color_b=(150, 70, 60), trail_color=(0,0,255))
    bot2 = Bot(name="blueBot", color_a=(106, 40, 30), color_b=(150, 70, 60), trail_color=(255,128,0))
    bot3 = Bot(name="yellowBot", color_a=(150, 106, 30), color_b=(150, 150, 60), trail_color=(0,255,255))

    # Create serial manager
    serial_manager = SerialManager()
    serial_manager.connect_serial(baud_rate=57600, port="/dev/ttyACM0")

    # Create video tracker with serial manager
    vt = AmakerBotTracker(camera_index=4, bot_trackers=[bot1, bot2], serial_manager=serial_manager)
    vt.start_tracking(recording=True)
