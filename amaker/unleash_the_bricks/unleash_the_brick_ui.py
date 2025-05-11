import logging
from typing import Callable
import datetime
import cv2
import numpy as np

WINDOW_TITLE = "aMaker microbot tracker"
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080

UI_VIDEO_TITLE = "Mission Unleash The Bricks"
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_BLACK = (0, 0, 0)
UI_TEXT_SCALE = 1.2
UI_TEXT_THICKNESS = 2
UI_DEFAULT_IMAGE_MASK = '/home/taccart/VSCode/amaker-path-capture/resources/backgroundmask_1920x1080.svg'
# UI constants
UI_BUTTON_COLOR_PRIMARY = (250, 250, 250)  # Button fill color
UI_BUTTON_COLOR_SECONDARY = (100, 100, 100)  # Button border color
UI_BUTTON_HEIGHT = 50
UI_BUTTON_WIDTH = 140
UI_BUTTON_X_POS = 1400
UI_BUTTON_MARGIN_X = 5
UI_BUTTON_MARGIN_Y = 10
UI_BUTTON_PADDING = 10
UI_BUTTON_SPACING = 40


# UI Messages


class AmakerUnleashTheBrickUI():
    # Define buttons
    def __init__(self, config: dict, on_click_start: Callable[[], None], on_click_stop: Callable[[], None],
                 on_click_safety: Callable[[], None], max_logs=10):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("UnleashTheBrickUI initialized with config: %s", config)
        self.config = config

        self.ui_button_start = {
            'x': UI_BUTTON_X_POS + UI_BUTTON_MARGIN_X
            , 'y': UI_BUTTON_MARGIN_Y
            , 'w': UI_BUTTON_WIDTH
            , 'h': UI_BUTTON_HEIGHT
            , 'text': 'Start'
            , 'clicked': False}
        self.ui_button_stop = {'x': UI_BUTTON_X_POS + UI_BUTTON_MARGIN_X + UI_BUTTON_WIDTH + UI_BUTTON_SPACING,
                               'y': UI_BUTTON_MARGIN_Y,
                               'w': UI_BUTTON_WIDTH, 'h': UI_BUTTON_HEIGHT, 'text': 'Stop', 'clicked': False}
        self.ui_button_safety = {'x': UI_BUTTON_X_POS + UI_BUTTON_MARGIN_X + 2 * (UI_BUTTON_WIDTH + UI_BUTTON_SPACING),
                                 'y': UI_BUTTON_MARGIN_Y, 'w': UI_BUTTON_WIDTH, 'h': UI_BUTTON_HEIGHT, 'text': 'Safety',
                                 'clicked': False}

        self.on_click_start = on_click_start
        self.on_click_stop = on_click_stop
        self.on_click_safety = on_click_safety
        self.is_fullscreen = False
        if config:
            if "fullscreen" in config:
                self.is_fullscreen = config["fullscreen"]


        self.logs = []
        self.max_logs = max_logs

    def add_log(self, message):
        current_time = datetime.datetime.now().strftime("%H%M%S")
        self.logs.append(current_time + " " + str(message))
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def draw_logs(self, frame):
        if self.logs:
            for i, log in enumerate(reversed(self.logs)):
                y_pos = frame.shape[0] - 30 - (i * 20)
                cv2.putText(frame, log, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def add_button(self, button, frame):
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_BUTTON_COLOR_PRIMARY, -1)  # Filled rectangle
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_BUTTON_COLOR_SECONDARY, 2)  # Border
        cv2.putText(frame, button['text'], (button['x'] + UI_BUTTON_PADDING, button['y'] + 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    UI_TEXT_SCALE, TEXT_COLOR_BLACK, UI_TEXT_THICKNESS)

    def overlay_mask_image(self, background, overlay, x, y):
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
            background_area = background[y:y + h, x:x + w, :3] * (1 - alpha)

            # Combine foreground and background
            background[y:y + h, x:x + w, :3] = foreground + background_area

    def initialize_window(self, video_capture, window_name, on_start, on_stop, on_safety):

        ret, input_frame = video_capture.read()
        if not ret:
            raise Exception("Failed to get video capture.")

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
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_OPENGL, 0)  # Disable OpenGL status bar
        cv2.resizeWindow(window_name, original_width, original_height)  # Start with original frame size
        self.add_button(self.ui_button_start, input_frame)
        self.add_button(self.ui_button_stop, input_frame)
        self.add_button(self.ui_button_safety, input_frame)
        return (input_frame, original_height, original_width)

    def set_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.UI_BUTTON_safety['x'] <= x <= self.UI_BUTTON_safety['x'] + self.UI_BUTTON_safety['w'] and
                    self.UI_BUTTON_safety['y'] <= y <=
                    self.UI_BUTTON_safety['y'] + self.UI_BUTTON_safety['h']):
                self.on_click_safety()  # Custom method to handle button click
            elif (self.UI_BUTTON_start['x'] <= x <= self.UI_BUTTON_start['x'] + self.UI_BUTTON_start['w'] and
                  self.UI_BUTTON_start['y'] <= y <=
                  self.UI_BUTTON_start['y'] + self.UI_BUTTON_start['h']):
                self.on_click_start()
            elif (self.UI_BUTTON_stop['x'] <= x <= self.UI_BUTTON_stop['x'] + self.UI_BUTTON_stop['w'] and
                  self.UI_BUTTON_stop['y'] <= y <=
                  self.UI_BUTTON_stop['y'] + self.UI_BUTTON_stop['h']):
                self.on_click_stop

        # Set the mouse callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        return [self.UI_BUTTON_start, self.UI_BUTTON_stop, self.UI_BUTTON_safety]

    def build_display_frame(self, input_frame):
        """Build the display frame with UI elements"""
        display_frame = input_frame.copy()  # Make a copy for display
        if self.is_fullscreen:
            display_frame = cv2.resize(display_frame, (self.screen_width, self.screen_height))
        else:
            # Use current window dimensions or default to original frame size
            display_frame = cv2.resize(display_frame, (self.window_width, self.window_height))

        cv2.putText(display_frame, UI_VIDEO_TITLE, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, TEXT_COLOR_WHITE, 5,
                    cv2.LINE_AA)

        # Load transparent PNG image (do this once in __init__ for efficiency)
        if not hasattr(self, 'logo_image'):
            # Load with transparency preserved
            self.logo_image = cv2.imread(UI_DEFAULT_IMAGE_MASK, cv2.IMREAD_UNCHANGED)
            # Optional  : resize it ?
            # self.logo_image = cv2.resize(self.logo_image, (width, height))

        # Add the transparent image to display_frame
        if hasattr(self, 'logo_image') and self.logo_image is not None:
            # Position in top-right corner (adjust x,y as needed)
            x = display_frame.shape[1] - self.logo_image.shape[1] - 20
            y = 20
            self.overlay_mask_image(display_frame, self.logo_image, x, y)

        # Your existing c   ode for buttons, text, etc.

        return display_frame
