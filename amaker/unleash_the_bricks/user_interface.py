import logging
from typing import Callable
from PIL import ImageFont, ImageDraw, Image
import os
import cv2
import numpy as np

from amaker.unleash_the_bricks import UI_BGRCOLOR_BLACK, UI_BGRCOLOR_GREY_DARK, UI_BGRCOLOR_WHITE, \
    UI_BGRCOLOR_BLUE_DARK, DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT

WINDOW_TITLE = "aMaker microbot tracker"

UI_VIDEO_TITLE = "Mission Unleash The Bricks"
UI_TITLE_COLOR = UI_BGRCOLOR_BLUE_DARK
UI_TITLE_FONT_SCALE = 40
UI_TITLE_FONT_NAME = "rubik"
UI_LOG_COLOR = UI_BGRCOLOR_BLACK
UI_LOG_FONT_SIZE =6
UI_LOG_FONT_NAME ="mono+b"


UI_BUTTON_SCALE = 0.6
UI_BUTTON_THICKNESS = 1
UI_BUTTON_TEXT_COLOR = UI_BGRCOLOR_BLACK
UI_DEFAULT_IMAGE_MASK = 'backgroundmask_1920x1080.jpg'
# UI constants
UI_BUTTON_COLOR_PRIMARY = UI_BGRCOLOR_WHITE  # Button fill color
UI_BUTTON_COLOR_SECONDARY = UI_BGRCOLOR_GREY_DARK  # Button border color
UI_BUTTON_HEIGHT = 40
UI_BUTTON_WIDTH = 75
UI_BUTTON_X_POS = 1000
UI_BUTTON_Y_POS = 10
UI_BUTTON_MARGIN_X = 5
UI_BUTTON_MARGIN_Y = 10
UI_BUTTON_PADDING = 10
UI_BUTTON_SPACING = 20

# UI Messages


class AmakerUnleashTheBrickGUI:

    def __init__(self, config: dict, buttons:dict[str, Callable], max_logs=10):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("UnleashTheBrickUI initialized with config: %s", config)
        self.config = config
        self.window_width=None
        self.window_height=None
        self.screen_width=None
        self.screen_height=None
        self.ui_buttons={}
        self.logo_image = None
        self.scaled_logo = None
        self.logo_loaded = False
        self._load_logo()
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
        self.ui_fonts_path={
            "mono+b":"/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
            "mono+bi":"/usr/share/fonts/truetype/freefont/FreeMonoBoldOblique.ttf",
            "mono+i":"/usr/share/fonts/truetype/freefont/FreeMonoOblique.ttf",
            "mono":"/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            "text+bi":"/usr/share/fonts/opentype/amadeus/Amadeus-BoldItalic.otf",
            "text+b":"/usr/share/fonts/opentype/amadeus/Amadeus-Bold.otf",
            "text+i":"/usr/share/fonts/opentype/amadeus/Amadeus-Italic.otf",
            "text":"/usr/share/fonts/opentype/amadeus/Amadeus-Regular.otf",
            "rubik":"/home/taccart/.local/share/fonts/RubikDistressed-Regular.ttf"}
        # Check if the font files exist
        for font_name, font_path in self.ui_fonts_path.items():
            if not os.path.exists(font_path):
                self.logger.warning(f"Font file {font_path} does not exist. Falling back to OpenCV font.")
                del self.ui_fonts_path[font_name]

    def _load_logo(self):
        self.logo_image = cv2.imread(UI_DEFAULT_IMAGE_MASK, cv2.IMREAD_UNCHANGED)
        if self.logo_image is None:
            self.logger.error(f"Failed to load mask image from {UI_DEFAULT_IMAGE_MASK}")
        else:
            self.logo_loaded = True

    def _resize_logo(self, target_width):
        if not self.logo_loaded or self.logo_image is None:
            return
        logo_scale_factor = target_width / DEFAULT_SCREEN_WIDTH
        if logo_scale_factor != 1.0:
            logo_width = int(self.logo_image.shape[1] * logo_scale_factor)
            logo_height = int(self.logo_image.shape[0] * logo_scale_factor)
            self.scaled_logo = cv2.resize(self.logo_image, (logo_width, logo_height))
        else:
            self.scaled_logo = self.logo_image

    def put_text_ttf(self, img, text, position, font_name, font_size, font_color):
        """
        Draw text on an image using a TTF font. If the font is not available, fall back to OpenCV's default font.
        Returns the gereated image so that it can be used in the pipeline.
        :param img:
        :param text:
        :param position:
        :param font_name:
        :param font_size:
        :param font_color:
        :return:
        """

        if not (font_name in self.ui_fonts_path):
            # Fall back to OpenCV font if TTF not available
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        1, font_color, thickness=1)
            return img
        else:
            # Load the font
            font_path = self.ui_fonts_path[font_name]
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception as e:
                self.logger.error(f"Failed to load font {font_name} from {font_path}: {e}")
                return img

            # Create a PIL image from the OpenCV image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # Draw the text
            draw.text(position, text, font=font, fill=font_color)

            # Convert back to OpenCV format
            result=cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            img[:]=result
            return img





    # def add_log(self, message):
    #     current_time = datetime.datetime.now().strftime("%H%M%S")
    #     self.logs.append(current_time + " " + str(message))
    #     if len(self.logs) > self.max_logs:
    #         self.logs.pop(0)
    #
    # def draw_logs(self, frame):
    #     if self.logs:
    #         for i, log in enumerate(reversed(self.logs)):
    #             y_pos = frame.shape[0] - 30 - (i * 16)
    #             self.put_text_ttf(frame, log, (10, y_pos), "mono", UI_LOG_FONTSCALE, UI_LOG_COLOR)

    def _add_button(self, button, frame):
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_BUTTON_COLOR_PRIMARY, -1)  # Filled rectangle
        cv2.rectangle(frame, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_BUTTON_COLOR_SECONDARY, 2)  # Border
        cv2.putText(frame, button['text'], (button['x'] + UI_BUTTON_PADDING, button['y'] + 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    UI_BUTTON_SCALE, UI_BUTTON_TEXT_COLOR, UI_BUTTON_THICKNESS)

    def _overlay_mask_image(self, background, overlay, x, y):
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



    def overlay_textlines(self, frame, text_lines:list[str], pos_x:int, pos_y:int, y_delta,reverse_lines:bool=False, font_name:str="text", font_size:int=10, font_color=(255, 255, 255)):
        """

        :param frame:
        :param text_lines: list of lines to write
        :param pos_x: if x>= 0,  0+x will be used as starting position else we start a window right border+x
        :param pos_y: if y>= 0,  0+x will be used as starting position else we start a window bottom +y
        :param y_delta: delta between lines
        :param reverse_lines: reverse lines
        :param font_name: font name
        :param font_size: font size
        :param font_color: font color
        :return:
        """
        lines=text_lines
        start_x=pos_x
        start_y=pos_y
        if pos_x<0:
            start_x=frame.shape[1] + pos_x
        if pos_y<0:
            start_y=frame.shape[0] + pos_y

        if reverse_lines:
            lines=reversed(lines)

        for i, line in enumerate(lines):
            real_start_y = start_y + i * y_delta
            frame=self.put_text_ttf(frame, line, (start_x, real_start_y), font_name, font_size, font_color)

        return frame


    def initialize_window(self, video_capture, window_name):
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
            self.logger.info(f"Detected screen size: {self.screen_width}x{self.screen_height}")
        except ImportError:
            self.logger.warning("screeninfo package not installed. Using default screen size.")
            self.screen_width, self.screen_height = DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT
        except Exception as e:
            self.screen_width, self.screen_height = DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT
            self.logger.warning(f"Couldn't detect screen size. Using defaults: {self.screen_width}x{self.screen_height} ({e})")

        cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_OPENGL, 1)  # Disable OpenGL status bar
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window_name, self._mouse_callback)

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
                if (value['x'] <= original_x <= value['x'] + value['w']
                        and value['y'] <= original_y <= value['y'] + value['h']):
                    value['action']()  # Fixed variable name
                    logging.info(f"Button clicked: {value['text']}", )


    def build_display_frame(self, input_frame):
        # display_frame = input_frame

        # Get original dimensions
        original_height, original_width = input_frame.shape[:2]
        _, _, window_width, window_height = cv2.getWindowImageRect(WINDOW_TITLE)

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
        cv2.resize(input_frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Add title with scaled parameters
        title_x = int(10 * scale_factor)
        title_y = int(10 * scale_factor)

        input_frame=self.put_text_ttf(img=input_frame, text=UI_VIDEO_TITLE, position=(title_x, title_y), font_name="rubik", font_size=UI_TITLE_FONT_SCALE, font_color=UI_TITLE_COLOR)

        # Use the pre-loaded logo image
        if self.logo_loaded and self.logo_image is not None:
            # Resize the logo if needed (or use pre-scaled version)
            if self.scaled_logo is None:
                self._resize_logo(target_width)

            if self.scaled_logo is not None:
                # Position in top-right corner
                margin = int(20 * scale_factor)
                x = input_frame.shape[1] - self.scaled_logo.shape[1] - margin
                y = margin
                self._overlay_mask_image(input_frame, self.scaled_logo, x, y)

        # Add buttons if provided
        for button in self.ui_buttons.values():
            self._add_button(button, input_frame)


