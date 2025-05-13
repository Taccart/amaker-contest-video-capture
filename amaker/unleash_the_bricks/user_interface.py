import datetime
import logging
from typing import Callable

import cv2
import numpy as np

WINDOW_TITLE = "aMaker microbot tracker"
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080

UI_VIDEO_TITLE = "Mission Unleash The Bricks"
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT_GRAY = (200, 200, 200)
COLOR_DARK_GRAY = (100, 100, 100)
COLOR_BLACK = (0, 0, 0)
UI_TITLE_COLOR = COLOR_WHITE
UI_TITLE_FONTSCALE = 1
UI_TITLE_THICKNESS = 2
UI_LOG_COLOR = COLOR_LIGHT_GRAY
UI_LOG_FONTSCALE = 0.5
UI_LOG_THICKNESS = 1

UI_BUTTON_SCALE = 0.6
UI_BUTTON_THICKNESS = 1
UI_BUTTON_TEXT_COLOR = COLOR_BLACK
UI_DEFAULT_IMAGE_MASK = 'backgroundmask_1920x1080.jpg'
# UI constants
UI_BUTTON_COLOR_PRIMARY = COLOR_LIGHT_GRAY  # Button fill color
UI_BUTTON_COLOR_SECONDARY = COLOR_BLACK  # Button border color
UI_BUTTON_HEIGHT = 40
UI_BUTTON_WIDTH = 75
UI_BUTTON_X_POS = 1000
UI_BUTTON_Y_POS = 10
UI_BUTTON_MARGIN_X = 5
UI_BUTTON_MARGIN_Y = 10
UI_BUTTON_PADDING = 10
UI_BUTTON_SPACING = 20


# UI Messages


class AmakerUnleashTheBrickGUI():
    # Define buttons
    def __init__(self, config: dict, buttons:dict[str, Callable], max_logs=10):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("UnleashTheBrickUI initialized with config: %s", config)
        self.config = config
        self.ui_buttons={}
        i=0
        self.is_fullscreen = False
        if config:
            if "fullscreen" in config:
                self.is_fullscreen = config["fullscreen"]


        self.logs = []
        self.max_logs = max_logs
        for key, value in buttons.items():
            if not  callable(value):
                raise ValueError(f"Button action for {key} is not callable.")

            self.ui_buttons.update ({key:{  'x': UI_BUTTON_X_POS + UI_BUTTON_MARGIN_X + i* (UI_BUTTON_WIDTH + UI_BUTTON_SPACING)
                , 'y': UI_BUTTON_Y_POS + UI_BUTTON_MARGIN_Y
                , 'w': UI_BUTTON_WIDTH
                , 'h': UI_BUTTON_HEIGHT
                , 'text': key
                , 'clicked': False
                , 'action' :value}})
            i+=1
    def add_log(self, message):
        current_time = datetime.datetime.now().strftime("%H%M%S")
        self.logs.append(current_time + " " + str(message))
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def draw_logs(self, frame):
        if self.logs:
            for i, log in enumerate(reversed(self.logs)):
                y_pos = frame.shape[0] - 30 - (i * 20)
                cv2.putText(frame, log, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, UI_LOG_FONTSCALE, UI_LOG_COLOR,
                            UI_LOG_THICKNESS)

    def add_button(self, button, frame):
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_BUTTON_COLOR_PRIMARY, -1)  # Filled rectangle
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_BUTTON_COLOR_SECONDARY, 2)  # Border
        cv2.putText(frame, button['text'], (button['x'] + UI_BUTTON_PADDING, button['y'] + 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    UI_BUTTON_SCALE, UI_BUTTON_TEXT_COLOR, UI_BUTTON_THICKNESS)

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
        self.on_click_safety = on_safety
        self.on_click_start = on_start
        self.on_click_stop = on_stop
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

        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO)

        # cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_OPENGL, 0)  # Disable OpenGL status bar
        cv2.setMouseCallback(window_name, self._mouse_callback)
        # cv2.resizeWindow(window_name, original_width, original_height)  # Start with original frame size

        return input_frame, original_height, original_width

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Correct variable names and add scaling compensation
            scale_x = 1.0
            scale_y = 1.0

            if self.is_fullscreen:
                # Scale mouse coordinates based on display size
                scale_x = self.window_width / self.screen_width
                scale_y = self.window_height / self.screen_height

            # Apply scaling to get original coordinates
            original_x = int(x * scale_x)
            original_y = int(y * scale_y)
            for key, value in self.ui_buttons.items():
                if (value['x'] <= original_x <= value['x'] + value['w'] and value['y'] <= original_y <= value['y'] + value['h']):
                    value['action']()  # Fixed variable name
                    logging.info(f"Button clicked: {value['text']}", )


    def build_display_frame(self, input_frame, buttons=None):
        """Build the display frame with UI elements"""
        display_frame = input_frame.copy()  # Make a copy for display

        # Get original dimensions
        original_height, original_width = input_frame.shape[:2]

        # Determine target dimensions
        if self.is_fullscreen:
            target_width, target_height = self.screen_width, self.screen_height
        else:
            target_width, target_height = self.window_width, self.window_height

        # Calculate scaling factors
        width_scale = target_width / original_width
        height_scale = target_height / original_height
        scale_factor = min(width_scale, height_scale)  # Use minimum to preserve aspect ratio

        # Resize frame
        display_frame = cv2.resize(display_frame, (target_width, target_height))

        # Scale UI elements
        font_scale = max(1, 1 * scale_factor)  # Minimum font scale 1
        text_thickness = max(2, int(1 * scale_factor))  # Minimum thickness 2

        # Add title with scaled parameters
        title_x = int(10 * scale_factor)
        title_y = int(30 * scale_factor)
        cv2.putText(display_frame, UI_VIDEO_TITLE, (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, UI_TITLE_FONTSCALE, UI_TITLE_COLOR,
                    UI_TITLE_THICKNESS, cv2.LINE_AA)

        # Load transparent PNG image (do this once in __init__ for efficiency)
        if not hasattr(self, 'logo_image'):
            # Load with transparency preserved
            self.logo_image = cv2.imread(UI_DEFAULT_IMAGE_MASK, cv2.IMREAD_UNCHANGED)
            if self.logo_image is None:
                self.logger.error(f"Failed to load mask image from {UI_DEFAULT_IMAGE_MASK}")

        # Add the transparent image to display_frame with scaling
        if hasattr(self, 'logo_image') and self.logo_image is not None:
            # Scale logo based on display size
            logo_scale_factor = target_width / DEFAULT_SCREEN_WIDTH
            logo_width = int(self.logo_image.shape[1] * logo_scale_factor)
            logo_height = int(self.logo_image.shape[0] * logo_scale_factor)

            # Only resize if necessary
            scaled_logo = cv2.resize(self.logo_image,
                                     (logo_width, logo_height)) if logo_scale_factor != 1.0 else self.logo_image

            # Position in top-right corner (adjust x,y as needed)
            margin = int(20 * scale_factor)
            x = display_frame.shape[1] - scaled_logo.shape[1] - margin
            y = margin
            self.overlay_mask_image(display_frame, scaled_logo, x, y)

        # Add buttons if provided
        for button in self.ui_buttons.values():
            self.add_button(button, display_frame)

        # Add logs with scaling
        if hasattr(self, 'logs'):
            self.draw_logs(display_frame)

        return display_frame
