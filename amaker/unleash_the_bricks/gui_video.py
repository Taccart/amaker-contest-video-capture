from datetime import datetime, timedelta
import logging
from typing import Callable

from PIL import ImageFont, ImageDraw, Image
import os
import cv2
import numpy as np

from amaker.unleash_the_bricks import UI_RGBCOLOR_BLACK, UI_RGBCOLOR_GREY_DARK, UI_RGBCOLOR_WHITE, \
    UI_RGBCOLOR_BLUE_DARK, DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT, \
    WINDOW_TITLE, UI_VIDEO_TITLE, UI_RGBCOLOR_ORANGE, UI_RGBCOLOR_RED, UI_RGBCOLOR_YELLOW
from amaker.unleash_the_bricks.bot import UnleashTheBrickBot



UI_BOT_INFO_FONT_COLOR = UI_RGBCOLOR_WHITE
UI_BOT_INFO_FONT_NAME = "Doto-Bold"
UI_BOT_INFO_FONT_SIZE = 30
UI_BOT_INFO_X = 5
UI_BOT_INFO_Y = -35
UI_BOT_INFO_Y_DELTA = -25

UI_COUNT_DOWN_X= -190
UI_COUNT_DOWN_Y= -70
UI_COUNT_DOWN_FONT_NAME="Doto-Bold"
UI_COUNT_DOWN_FONT_SIZE=60
UI_COUNT_DOWN_COLOR_LONG=UI_RGBCOLOR_YELLOW
UI_COUNT_DOWN_COLOR_MEDIUM=UI_RGBCOLOR_ORANGE
UI_COUNT_DOWN_COLOR_SHORT=UI_RGBCOLOR_RED
UI_COUNT_DOWN_LONG_MINUTES=2
UI_COUNT_DOWN_MEDIUM_MINUTES=1
UI_BOT_TAG_FONT_NAME="Doto-Bold"
UI_BOT_TAG_FONT_SIZE=20
UI_BOT_TRAIL_WIDTH = 2
UI_BUTTON_COLOR_PRIMARY = UI_RGBCOLOR_WHITE  # Button fill color
UI_BUTTON_COLOR_SECONDARY = UI_RGBCOLOR_GREY_DARK  # Button border color
UI_BUTTON_FONT_COLOR = UI_RGBCOLOR_BLUE_DARK
UI_BUTTON_FONT_NAME = "Doto-Bold"
UI_BUTTON_FONT_SIZE = 14
UI_BUTTON_HEIGHT = 40 # Button  height
UI_BUTTON_MARGIN_X = 5
UI_BUTTON_MARGIN_Y = 10
UI_BUTTON_PADDING = 10
UI_BUTTON_SCALE = 0.6
UI_BUTTONS_DIRECTION = "horizontal"  # Button layout direction
UI_BUTTON_SPACING = 20
UI_BUTTON_TEXT_COLOR = UI_RGBCOLOR_BLACK
UI_BUTTON_THICKNESS = 1
UI_BUTTON_WIDTH = 75
UI_BUTTON_X_POS = 1000
UI_BUTTON_Y_POS = 10
UI_DEFAULT_IMAGE_MASK = 'backgroundmask_1920x1080.jpg'
UI_LOG_COLOR = UI_RGBCOLOR_BLACK
UI_LOG_FONT_COLOR = UI_RGBCOLOR_WHITE
UI_LOG_FONT_NAME ="CutiveMono"
UI_LOG_FONT_SIZE=10
UI_LOG_X=10
UI_LOG_Y=-20
UI_LOG_Y_DELTA=-16
UI_TITLE_COLOR = UI_RGBCOLOR_BLUE_DARK
UI_TITLE_FONT_NAME="RubikMoonrocks"
UI_TITLE_FONT_SCALE = 40


# UI Messages


class AmakerUnleashTheBrickVideo:
    """
    Class managing the all the graphic user interface for aMaker microbot tounrament 2025 "Unleash The Bricks"
    """

    def __init__(self, config: dict, buttons:dict[str, Callable], max_logs=10):
        self._LOG = logging.getLogger(__name__)
        self.is_overlay_logs=False

        self._LOG.debug("UnleashTheBrickUI initialized with config: %s", config)
        self.config = config
        self.window_width=None
        self.window_height=None
        self.screen_width=None
        self.screen_height=None
        self.ui_buttons={}

        self.ui_buttons = {}
        self.logo_image = None
        self.scaled_logo = None
        self.logo_loaded = False
        self._load_logo()
        fonts_package = "amaker.unleash_the_bricks.resources.fonts"
        fonts_dir =self.get_fonts_directory()
        i=0
        self.is_fullscreen = False
        if config:
            if "fullscreen" in config:
                self.is_fullscreen = config["fullscreen"]


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
        # Load fonts. This should really be improved.
        self.ui_fonts_path = {
            "Doto-Black": os.path.join(fonts_dir, "Doto", "static/Doto-Black.ttf"),
            "Doto_Rounded-Bold": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-Bold.ttf"),
            "Doto-Thin": os.path.join(fonts_dir, "Doto", "static/Doto-Thin.ttf"),
            "Doto_Rounded-Light": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-Light.ttf"),
            "Doto_Rounded-Thin": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-Thin.ttf"),
            "Doto-ExtraBold": os.path.join(fonts_dir, "Doto", "static/Doto-ExtraBold.ttf"),
            "Doto_Rounded-Black": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-Black.ttf"),
            "Doto_Rounded-SemiBold": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-SemiBold.ttf"),
            "Doto_Rounded": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-Regular.ttf"),
            "Doto-SemiBold": os.path.join(fonts_dir, "Doto", "static/Doto-SemiBold.ttf"),
            "Doto_Rounded-ExtraLight": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-ExtraLight.ttf"),
            "Doto_Rounded-ExtraBold": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-ExtraBold.ttf"),
            "Doto-Medium": os.path.join(fonts_dir, "Doto", "static/Doto-Medium.ttf"),
            "Doto_Rounded-Medium": os.path.join(fonts_dir, "Doto", "static/Doto_Rounded-Medium.ttf"),
            "Doto": os.path.join(fonts_dir, "Doto", "static/Doto-Regular.ttf"),
            "Doto-ExtraLight": os.path.join(fonts_dir, "Doto", "static/Doto-ExtraLight.ttf"),
            "Doto-Bold": os.path.join(fonts_dir, "Doto", "static/Doto-Bold.ttf"),
            "Doto-Light": os.path.join(fonts_dir, "Doto", "static/Doto-Light.ttf"),
            "Doto-VariableFont_ROND,wght": os.path.join(fonts_dir, "Doto", "Doto-VariableFont_ROND,wght.ttf"),

            "CutiveMono": os.path.join(fonts_dir, "Cutive_Mono", "CutiveMono-Regular.ttf"),

            "RubikMoonrocks": os.path.join(fonts_dir, "Rubik_Moonrocks", "RubikMoonrocks-Regular.ttf"),

        }
        kept_fonts= {}
        for font_name, font_path in self.ui_fonts_path.items():
            if  os.path.exists(font_path):
                kept_fonts[font_name]= font_path
            else:
                self._LOG.warning(f"Font not found: {font_path}")
        self.ui_fonts_path=kept_fonts


    def get_fonts_directory(self):
        """Get the fonts directory in a way that works with regular installs and egg installs"""
        try:
            # First attempt: Use importlib.resources (Python 3.7+)
            import importlib.resources
            try:
                # For Python 3.9+
                with importlib.resources.files('amaker.unleash_the_bricks.resources') as p:
                    fonts_dir = p / 'fonts'
                    if fonts_dir.exists():
                        return str(fonts_dir)
            except AttributeError:
                # For Python 3.7-3.8 - not tested
                try:
                    with importlib.resources.path('amaker.unleash_the_bricks.resources', 'fonts') as p:
                        if p.exists():
                            return str(p)
                except Exception as e:
                    self._LOG.warning(f"Could not find fonts using importlib.resources.path: {e}")
                    pass
        except ImportError:
            pass

        # Second attempt: Get the directory from the current file path
        try:
            current_file = os.path.abspath(__file__)
            package_dir = os.path.dirname(current_file)
            fonts_dir = os.path.join(package_dir, 'resources', 'fonts')
            if os.path.isdir(fonts_dir):
                return fonts_dir
        except Exception:
            pass

        # Last resort: look for a relative path
        relative_fonts_dir = os.path.join('amaker', 'unleash_the_bricks', 'resources', 'fonts')
        if os.path.isdir(relative_fonts_dir):
            return relative_fonts_dir

        self._LOG.warning("Could not locate fonts directory, using current directory")
        return "."

    def _load_logo(self):
        pass
        # self.logo_image = cv2.imread(UI_DEFAULT_IMAGE_MASK, cv2.IMREAD_UNCHANGED)
        # if self.logo_image is None:
        #     self.logger.error(f"Failed to load mask image from {UI_DEFAULT_IMAGE_MASK}")
        # else:
        #     self.logo_loaded = True

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
        x=position[0]
        y=position[1]
        if x < 0:

            x= img.shape[1] + x
        if y<0:
            y=img.shape[0] + y
        if not (font_name in self.ui_fonts_path):
            # Fall back to OpenCV font if TTF not available
            cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size/20, font_color, thickness=1)
            return img
        else:
            # Load the font
            font_path = self.ui_fonts_path[font_name]
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception as e:
                self._LOG.error(f"Failed to load font {font_name} from {font_path}: {e}")
                return img

            # Create a PIL image from the OpenCV image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # Draw the text
            draw.text((x,y), text, font=font, fill=font_color)

            # Convert back to OpenCV format
            result=cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            img[:]=result
            return img

    def ui_add_button(self, button, img):
        # cv2.rectangle(img, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
        #               UI_BUTTON_COLOR_PRIMARY, -1)  # Filled rectangle
        cv2.rectangle(img, (button['x'], button['y']), (button['x'] + button['w'], button['y'] + button['h']),
                      UI_BUTTON_COLOR_SECONDARY, 2)  # Border

        self.put_text_ttf(img=img,
                          text=button['text'],
                          position=(button['x'] + UI_BUTTON_PADDING, button['y'] + UI_BUTTON_HEIGHT/2),
                          font_name=UI_BUTTON_FONT_NAME, font_size=UI_BUTTON_FONT_SIZE, font_color=UI_BUTTON_FONT_COLOR)


    def ui_overlay_mask_image(self, background, overlay, x, y):
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

    def ui_add_countdown(self, img, deadline:datetime):
        """
        Overlay a countdown timer on the frame
        :param img:
        :param deadline:
        :return:
        """
        if deadline is None:
            return

        time_left = deadline - datetime.now()
        if time_left.total_seconds() <= 0:
            return img

        countdown_color=UI_COUNT_DOWN_COLOR_LONG if time_left> timedelta(minutes=UI_COUNT_DOWN_LONG_MINUTES) else UI_COUNT_DOWN_COLOR_MEDIUM if time_left> timedelta(minutes=UI_COUNT_DOWN_MEDIUM_MINUTES) else UI_COUNT_DOWN_COLOR_SHORT

        minutes, seconds = divmod(int(time_left.total_seconds()), 60)
        countdown_text = f"{minutes:02}:{seconds:02}"
        img=self.put_text_ttf(img=img,
                              text=countdown_text,
                              position=(UI_COUNT_DOWN_X , UI_COUNT_DOWN_Y),
                              font_name=UI_COUNT_DOWN_FONT_NAME,
                              font_size=UI_COUNT_DOWN_FONT_SIZE,
                              font_color=countdown_color)
        return img
    def ui_add_bot(self, amaker_ui, mtx, dist, img, bot: UnleashTheBrickBot):
        """
        Overlay a bot on the frame, with box, trail and head direction
        :param img:
        :param bot:
        :return
        """
        tag = bot.get_last_tag_position()
        if tag is None:
            return img

        # Convert corners to int once
        corners_int = tag.corners.astype(int)

        # Add text
        # img= amaker_ui.put_text_ttf(img,
        #                             text=bot.name,
        #                             position=(corners_int[0, 0] + 10, corners_int[0, 1] + 10),
        #                             font_name=UI_BOT_TAG_FONT_NAME,
        #                             font_size=UI_BOT_TAG_FONT_SIZE,
        #                             font_color=bot.color)
        brg_color=(bot.color[2],bot.color[1],bot.color[0])
        # Draw trail more efficiently
        trail = bot.get_trail()
        if trail and len(trail) > 1:
            # Pre-compute all centers at once
            centers = np.array([t.center.astype(int) for t in trail])
            # Draw lines in one loop
            for i in range(len(centers) - 1):
                cv2.line(img, tuple(centers[i]), tuple(centers[i + 1]), brg_color, UI_BOT_TRAIL_WIDTH)

        # Draw tag corners
        for idx in range(len(corners_int)):
            if idx!=2:
                cv2.line(img, tuple(corners_int[idx - 1]), tuple(corners_int[idx]), brg_color, 5)

        # Only compute 3D axes if needed and tag has pose information
        if hasattr(tag, 'pose_R') and hasattr(tag, 'pose_t'):
            # Define axis points outside the conditional to avoid recreating every time
            axis_length = 0.10  # 10cm axis length
            axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])

            # Pre-reshape pose_t
            pose_t = tag.pose_t.reshape(3, 1)

            # Create rotation vector once
            origin_vector, _ = cv2.Rodrigues(tag.pose_R)

            # Project points
            image_points, _ = cv2.projectPoints(axis_points, origin_vector, pose_t, mtx, dist)
            image_points = np.int32(image_points).reshape(-1, 2)

            # Draw only the necessary axis line
            origin = tuple(image_points[0])
            cv2.line(img, origin, tuple(image_points[1]), brg_color, 3)  # X-axis only

        return img


    def ui_add_tag(self, img, mtx, dist, tag, color=UI_RGBCOLOR_GREY_DARK, label=None, has_axis=False):
        """
        Overlay a tag on the frame
        :param img:
        :param tag:
        :param color:
        :param label:
        :param has_axis:
        :return:
        """
        for idx in range(len(tag.corners)):
            cv2.line(img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), color,
                     5)
        cv2.putText(img, label,
                    org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.8, color=color, thickness=2)

        if has_axis:
            # Draw the 3D axes on bot
            axis_length = 0.10  # 10cm axis length
            # Define the 3D points for axes (origin, x-axis, y-axis, z-axis)
            axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])

            # Debug - check if pose information is available
            if hasattr(tag, 'pose_R') and hasattr(tag, 'pose_t'):
                # Make sure pose_t is properly shaped
                pose_t = tag.pose_t.reshape(3, 1)

                # Create rotation and translation vectors for projection
                origin_vector, _ = cv2.Rodrigues(tag.pose_R)  # Convert rotation matrix to rotation vector
                destination_vector = pose_t

                # Project the 3D points directly with proper rvec and tvec
                image_points, _ = cv2.projectPoints(axis_points, origin_vector, destination_vector, mtx,
                                                    dist)
                image_points = np.int32(image_points).reshape(-1, 2)

                # Draw the axes with thicker lines and clear colors
                origin = tuple(image_points[0])
                cv2.line(img, origin, tuple(image_points[1]), color, 3)  # X-axis (red)
                cv2.line(img, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y-axis (green)
                cv2.line(img, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z-axis (blue)
                #Add labels to the axes for better visibility
                cv2.putText(img, "X", tuple(image_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img, "Y", tuple(image_points[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, "Z", tuple(image_points[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Log axis points for debugging
                # self._LOG.debug(f"Axis points: Origin={image_points[0]}, X={image_points[1]}, Y={image_points[2]}, Z={image_points[3]}")
        return img

    def ui_add_textlines(self, img, text_lines:list[str], pos_x:int, pos_y:int, y_delta, reverse_lines:bool=False, font_name:str= "text", font_size:int=10, font_color=(255, 255, 255)):
        """

        :param img:
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
            start_x= img.shape[1] + pos_x
        if pos_y<0:
            start_y= img.shape[0] + pos_y

        if reverse_lines:
            lines=reversed(lines)

        for i, line in enumerate(lines):
            real_start_y = start_y + i * y_delta
            img=self.put_text_ttf(img, line, (start_x, real_start_y), font_name, font_size, font_color)

        return img


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
            self._LOG.info(f"Detected screen size: {self.screen_width}x{self.screen_height}")
        except ImportError:
            self._LOG.warning("screeninfo package not installed. Using default screen size.")
            self.screen_width, self.screen_height = DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT
        except Exception as e:
            self.screen_width, self.screen_height = DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT
            self._LOG.warning(f"Couldn't detect screen size. Using defaults: {self.screen_width}x{self.screen_height} ({e})")

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
                    self._LOG.info(f"Button clicked: {value['text']}", )


    def build_display_frame(self, img):

        # Get original dimensions
        original_height, original_width = img.shape[:2]
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
        cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

        # Add title with scaled parameters
        title_x = int(10 * scale_factor)
        title_y = int(10 * scale_factor)

        img=self.put_text_ttf(img=img, text=UI_VIDEO_TITLE, position=(title_x, title_y), font_name=UI_TITLE_FONT_NAME, font_size=UI_TITLE_FONT_SCALE, font_color=UI_TITLE_COLOR)

        # Use the pre-loaded logo image
        if self.logo_loaded and self.logo_image is not None:
            # Resize the logo if needed (or use pre-scaled version)
            if self.scaled_logo is None:
                self._resize_logo(target_width)

            if self.scaled_logo is not None:
                # Position in top-right corner
                margin = int(20 * scale_factor)
                x = img.shape[1] - self.scaled_logo.shape[1] - margin
                y = margin
                self.ui_overlay_mask_image(img, self.scaled_logo, x, y)

        # Add buttons if provided
        for button in self.ui_buttons.values():
            self.ui_add_button(button, img)
        return img


    def show_logs(self, img, logs):
        if not(not (logs is None ) & self.is_overlay_logs):
            self.ui_add_textlines(img=img,
                                  text_lines=logs,
                                  pos_x=UI_LOG_X,
                                  pos_y=UI_LOG_Y,
                                  y_delta=UI_LOG_Y_DELTA,
                                  reverse_lines=True,
                                  font_name=UI_LOG_FONT_NAME,
                                  font_size=UI_LOG_FONT_SIZE,
                                  font_color=UI_LOG_FONT_COLOR,
                                  )


    def show_bot_infos(self, img, tracked_bots, ):
        """
        Overlay bot information on the frame
        :param img:
        :return:
        """
        i = 0
        for bot_id, bot in tracked_bots.items():
            y_pos = UI_BOT_INFO_Y + (i * UI_BOT_INFO_Y_DELTA)

            img= self.put_text_ttf(img=img,
                                        text=bot.name,
                                        position=(UI_BOT_INFO_X, y_pos),
                                        font_name=UI_BOT_INFO_FONT_NAME,
                                        font_size=UI_BOT_INFO_FONT_SIZE,
                                        font_color=bot.color)


            i += 1