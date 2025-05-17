import datetime
import logging
import os
import cv2
import numpy as np

from amaker.communication.communication_abstract import CommunicationManagerAbstract
# Import SerialManager from the new file
from amaker.communication.serial_communication_manager import SerialCommunicationManagerImpl
from amaker.detection.detector_apriltag import AprilTagDetectorImpl
from amaker.unleash_the_bricks import UI_BGRCOLOR_BLUE_DARK, UI_BGRCOLOR_GREEN_LIGHT, \
    UI_BGRCOLOR_GREY_LIGHT, UI_BGRCOLOR_BLUE_LIGHT, UI_BGRCOLOR_GREY_MEDIUM, UI_BGRCOLOR_WHITE, UI_BGRCOLOR_GREY_DARK, \
    DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT
from amaker.unleash_the_bricks.bot import UnleashTheBrickBot
from user_interface import AmakerUnleashTheBrickGUI

###
# microbit side:
# at start : button A to scroll radio group, A+B save radio group
#
# thread sends heartbeats every X seconds
# listen on channel
# lib => main() to s

# ===== Global Constants =====
# Environment settings
os.environ["QT_QPA_PLATFORM"] = "xcb"
# exit codes
EXIT_NO_CAMERA = 1
EXIT_INVALID_CAMERA_CHOICE = 2
# OpenCV constants
CV_THREADS = 6
# Window settings
# CAMERA_SEARCH_LIMIT

CAMERA_SEARCH_LIMIT = 10
WINDOW_TITLE = "aMaker microbot tracker"

UI_TITLE_COLOR = UI_BGRCOLOR_BLUE_DARK

UI_BOT_INFO_FONT_COLOR = UI_BGRCOLOR_BLUE_LIGHT
UI_BOT_INFO_FONT_SIZE = 30
UI_BOT_INFO_FONT_NAME = "mono+b"
UI_BOT_TRAIL_WIDTH = 2
UI_BOT_INFO_X = 10
UI_BOT_INFO_Y = 60
UI_BOT_INFO_Y_DELTA = 25
UI_COLOR_TAG_UNKNOWN = UI_BGRCOLOR_GREY_DARK
UI_COLOR_TAG_WALL = UI_BGRCOLOR_GREY_LIGHT
UI_COLOR_TAG_GROUND = UI_BGRCOLOR_GREY_MEDIUM
UI_LOG_FONT_COLOR = UI_BGRCOLOR_WHITE
UI_LOG_FONT_NAME = "mono+b"
UI_LOG_FONT_SIZE=10
UI_LOG_X=10
UI_LOG_Y_DELTA=10
UI_LOG_Y=-120
LOG_MAX_LINES=10
COLOR_TAG_IGNORED = UI_BGRCOLOR_GREY_MEDIUM
COLOR_TAG_WALL = UI_BGRCOLOR_GREY_MEDIUM
COLOR_TAG_GROUND = UI_BGRCOLOR_GREY_LIGHT
COLOR_TAG_BOT = UI_BGRCOLOR_GREEN_LIGHT
# Video recording constants
VIDEO_OUT_CODEC = 'XVID'
VIDEO_OUT_FPS = 30.0
VIDEO_OUT_WIDTH = 1280
VIDEO_OUT_HEIGHT = 720

APRILTAG_SIZE_M = 0.1  # Size of the AprilTag in meters
# Key codes
KEY_ESC = 27
KEY_CTRL_D = 4
KEY_Q = ord('q')
KEY_F = ord('f')

COMMAND_START = "START"
COMMAND_STOP = "STOP"
COMMAND_SAFETY = "SAFETY"


# Bot tracker constants


class AmakerBotTracker():

    def __init__(self, calibration_file, camera_index: int = 0, tracked_bots: dict[int, UnleashTheBrickBot] = None,
                 communication_manager=None, window_size=(DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT),
                 know_tags: dict[int, dict] = None):
        self.reference_tags = know_tags
        self.logs = []
        self.max_logs = LOG_MAX_LINES
        self.window_size = window_size
        self.calibration_file = calibration_file
        calibration_data = np.load(calibration_file)
        self.mtx = calibration_data['camera_matrix']
        self.dist = calibration_data['dist_coeffs']
        self.video_writer = None
        self.communication_manager = None
        if communication_manager:
            if isinstance(communication_manager, CommunicationManagerAbstract):
                self.communication_manager = communication_manager
            else:
                raise TypeError("communication_manager must be an instance of CommunicationManagerAbstract")

        if communication_manager:
            logging.info("Serial communication activated.")
        else:
            logging.info("Serial communication not activated.")
        self.communication_manager.register_on_data_callback(self._on_data_received)
        self.bot_tracker = AprilTagDetectorImpl(calibration_file=self.calibration_file)
        if isinstance(tracked_bots, dict):
            self.tracked_bots = tracked_bots
        else:
            logging.error(
                "Initialization error : tracked_bots must be a dictionary of int->UnleashTheBrickBot instances")
        self.camera_index = self.user_input_camera_choice() if camera_index < 0 else camera_index

        self.video_capture = cv2.VideoCapture(self.camera_index)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_OUT_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_OUT_HEIGHT)

        cv2.setNumThreads(CV_THREADS)  # Set the number of threads for OpenCV

        if not self.video_capture.isOpened():
            raise ValueError(f"Camera {camera_index} not found or cannot be opened.")
        self.amaker_ui = AmakerUnleashTheBrickGUI(config={}
                                                  , buttons={"start": self._on_UI_BUTTON_start
                , "stop": self._on_UI_BUTTON_stop
                , "safety": self._on_UI_BUTTON_safety})

    def _cleanup_resources(self):
        try:
            if self.communication_manager:
                self.communication_manager.close()
                logging.info("Communication closed.")
        except Exception as e:
            logging.warning(f"Error during communication cleanup: {e}")

        try:
            self.video_capture.release()
            logging.info("Video capture released.")
        except Exception as e:
            logging.warning(f"Error during video capture release: {e}")

        try:
            if self.video_writer and self.video_writer.isOpened():
                # Release the video writer
                self.video_writer.release()
                logging.info("Video writer released successfully.")
        except Exception as e:
            logging.warning(f"Error during video capture release: {e}")

    # Button callback functions
    def _on_UI_BUTTON_start(self):
        """Handle start button click"""

        if self.communication_manager:
            self.communication_manager.send(COMMAND_START)
            self._add_log(f"> {COMMAND_START} sent")
        else:
            self._add_log(f"! {COMMAND_START} failed to send")

    def _on_UI_BUTTON_stop(self):
        """Handle stop button click"""
        if self.communication_manager:
            self.communication_manager.send(COMMAND_STOP)
            self._add_log(f"> {COMMAND_STOP}")
        else:
            self._add_log(f"! {COMMAND_STOP} failed to send")

    def _on_UI_BUTTON_safety(self):
        """Handle safety button click"""
        if self.communication_manager:
            self.communication_manager.send(COMMAND_SAFETY)
            self._add_log(f"> {COMMAND_SAFETY}")
        else:
            self._add_log(f"! {COMMAND_SAFETY} failed to send")

    def _on_data_received(self, data):
        self._add_log("<" + data)

    def _add_log(self, message):
        """
        Add a log message to the logs list (keeping a limit on the number of lines)
        :param message:
        :return:
        """
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append(current_time + " " + str(message))
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def _overlay_bot(self, frame, bot: UnleashTheBrickBot):
        """
        Overlay a bot on the frame, with box, trail and head direction
        :param frame:
        :param bot:
        :return:
        """
        tag = bot.get_last_tag_position()
        if tag is None:
            return frame

        # Convert corners to int once
        corners_int = tag.corners.astype(int)

        # Add text
        frame=self.amaker_ui.put_text_ttf(frame,
                                            text=bot.name,
                                            position=(corners_int[0, 0] + 10, corners_int[0, 1] + 10),
                                            font_name="mono+b",
                                            font_size=10,
                                            font_color=bot.color)

        # Draw trail more efficiently
        trail = bot.get_trail()
        if trail and len(trail) > 1:
            # Pre-compute all centers at once
            centers = np.array([t.center.astype(int) for t in trail])
            # Draw lines in one loop
            for i in range(len(centers) - 1):
                cv2.line(frame, tuple(centers[i]), tuple(centers[i + 1]), bot.color, UI_BOT_TRAIL_WIDTH)

        # Draw tag corners
        for idx in range(len(corners_int)):
            cv2.line(frame, tuple(corners_int[idx - 1]), tuple(corners_int[idx]), bot.color, 5)

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
            image_points, _ = cv2.projectPoints(axis_points, origin_vector, pose_t, self.mtx, self.dist)
            image_points = np.int32(image_points).reshape(-1, 2)

            # Draw only the necessary axis line
            origin = tuple(image_points[0])
            cv2.line(frame, origin, tuple(image_points[1]), bot.color, 3)  # X-axis only

        return frame

    def _overlay_tag(self, frame, tag, color=UI_COLOR_TAG_UNKNOWN, label=None, has_axis=False):
        """
        Overlay a tag on the frame
        :param frame:
        :param tag:
        :param color:
        :param label:
        :param has_axis:
        :return:
        """
        for idx in range(len(tag.corners)):
            cv2.line(frame, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), color,
                     5)
        cv2.putText(frame, label,
                    org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=1)

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
                image_points, _ = cv2.projectPoints(axis_points, origin_vector, destination_vector, self.mtx,
                                                    self.dist)
                image_points = np.int32(image_points).reshape(-1, 2)

                # Draw the axes with thicker lines and clear colors
                origin = tuple(image_points[0])
                cv2.line(frame, origin, tuple(image_points[1]), color, 3)  # X-axis (red)
                cv2.line(frame, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y-axis (green)
                cv2.line(frame, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z-axis (blue)
                #Add labels to the axes for better visibility
                cv2.putText(frame, "X", tuple(image_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Y", tuple(image_points[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Z", tuple(image_points[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Log axis points for debugging
                # logging.debug(f"Axis points: Origin={image_points[0]}, X={image_points[1]}, Y={image_points[2]}, Z={image_points[3]}")
        return frame

    def _overlay_tags(self, frame, tags):
        """
        Overlay detected tags on the frame
        :param frame:
        :param tags:
        :return:
        """
        for tag in tags:

            color = UI_COLOR_TAG_UNKNOWN
            if tag.tag_id in self.tracked_bots.keys():
                self.tracked_bots[tag.tag_id].add_tag_position(tag)
                self._overlay_bot(frame, self.tracked_bots[tag.tag_id])

            elif tag.tag_id in self.reference_tags.keys():
                if self.reference_tags[tag.tag_id]["type"] == "wall":
                    color = UI_COLOR_TAG_WALL
                if self.reference_tags[tag.tag_id]["type"] == "ground":
                    color = UI_COLOR_TAG_GROUND
                self._overlay_tag(frame, tag, color=color)
            else:
                logging.debug(f"Tag {tag.tag_id} unknown : not shown. ")
            logging.debug(f"BOT {tag.tag_id},  T={tag.pose_t} R={tag.pose_R} C={tag.center}")


    def _overlay_logs(self, frame):
        if self.logs:
            self.amaker_ui.overlay_textlines(frame=frame,
                                                     text_lines=self.logs,
                                                     pos_x=UI_LOG_X,
                                                     pos_y=UI_LOG_Y,
                                                     y_delta=UI_LOG_Y_DELTA,
                                                     reverse_lines=False,
                                                     font_name=UI_LOG_FONT_NAME,
                                                     font_size=UI_LOG_FONT_SIZE,
                                                     font_color=UI_LOG_FONT_COLOR)


    def _overlay_bot_infos(self, frame):
        """
        Overlay bot information on the frame
        :param frame:
        :return:
        """
        i = 0
        for bot_id, bot in self.tracked_bots.items():
            y_pos = UI_BOT_INFO_Y + (i * UI_BOT_INFO_Y_DELTA)

            frame=self.amaker_ui.put_text_ttf(img=frame,
                                                text=bot.get_bot_info(),
                                                position=(UI_BOT_INFO_X, y_pos),
                                                font_name=UI_BOT_INFO_FONT_NAME,
                                                font_size=UI_BOT_INFO_FONT_SIZE,
                                                font_color=UI_BOT_INFO_FONT_COLOR)
            # cv2.putText(frame, bot.get_bot_info(),
            #             (UI_BOT_INFO_X, y_pos),
            #             UI_BOT_INFO_FONT, UI_BOT_INFO_FONT_SCALE,
            #             UI_BOT_INFO_FONT_COLOR, UI_BOT_INFO_FONT_THICKNESS)
            i += 1

    @staticmethod
    def user_input_camera_choice() -> int | None:
        """Select a camera from available cameras"""
        index = 0
        available_cameras = []
        logging.info(f"Searching for cameras from #0 to #{CAMERA_SEARCH_LIMIT} ...")
        cap = None
        while True:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    logging.info(f"Open camera {index} : failed.")
                else:
                    logging.info(f"Open camera {index} : succeeded.")
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
            print(f" - Camera {cam}")

        selected_camera = int(input("\nSelect the camera index to use: "))
        if selected_camera not in available_cameras:
            logging.error("Invalid camera index selected.")
            exit(EXIT_INVALID_CAMERA_CHOICE)
        else:
            return selected_camera

    def setup_video_recording(self, is_recording: bool, path: str):
        """Setup video recording"""
        if is_recording:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_OUT_CODEC)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f'{path}aMaker_microbot_tracker_{current_time}.avi'
            self.video_writer = cv2.VideoWriter(video_name, fourcc, VIDEO_OUT_FPS, (VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT))
            if self.video_writer.isOpened():
                logging.info(f"Video recording started: {video_name}")
            else:
                logging.error(f"Video recording not started: {video_name}")
        else:
            logging.info("No video recording.")

    def main_tracking_loop(self, window_name):
        while True:
            ret, input_frame = self.video_capture.read()
            #input_frame = cv2.resize(input_frame, self.window_size)
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break

            # Show the video feed
            detected_tags = self.bot_tracker.detect(input_frame)
            self._overlay_tags(input_frame, detected_tags)
            self._overlay_bot_infos(input_frame)
            self._overlay_logs(input_frame)

            self.amaker_ui.build_display_frame(input_frame)

            cv2.imshow(window_name, input_frame)
            if self.video_writer and self.video_writer.isOpened():
                # Save the frame to the video file
                video_out = cv2.resize(input_frame, (VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT))
                self.video_writer.write(video_out)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_Q or key == KEY_CTRL_D or key == KEY_ESC:  # 'q' or Ctrl+D or ESC to quit
                break

    def start_tracking(self, recording: bool = False, recording_path: str = ".", window_name=WINDOW_TITLE):
        """Start tracking bots in video feed"""
        input_frame, original_height, original_width = self.amaker_ui.initialize_window(self.video_capture, window_name
                                                                                        )
        logging.info(f"Bot trackers: {self.tracked_bots}")
        logging.info(f"Press 'q' or ESC to quit.")
        try:
            self.setup_video_recording(recording, path=recording_path)
            self.main_tracking_loop(window_name)
        except KeyboardInterrupt as e:
            logging.warning(f"Keyboard interruption.")
        except Exception as e:
            logging.error(f"Error during video tracking: {e}")
            raise e
        finally:
            self._cleanup_resources()

    def __del__(self):
        """Destructor cleans up resources"""
        try:
            if self.communication_manager:
                self.communication_manager.close()
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

    import argparse

    parser = argparse.ArgumentParser(description='Unleash the bricks: bots controller')

    parser.add_argument('--window_width', metavar='number', required=False,
                        help='Window width', type=int, default=DEFAULT_SCREEN_WIDTH)
    parser.add_argument('--window_height', metavar='number', required=False,
                        help='Window width', type=int, default=DEFAULT_SCREEN_HEIGHT)

    parser.add_argument('--camera_calibration_file', metavar='path', required=False,
                        help='Path to camera calibration file', type=str, default='camera_calibration.npz')
    parser.add_argument('--camera_number', metavar='number', required=True,
                        help='Camera number. Put -1 to choose in a generated list of accessible ones.', type=int,
                        default=-1)

    parser.add_argument('--serial_port', metavar='path', required=False,
                        help='Serial port (ex: /dev/ttyACM0)', type=str, default='/dev/ttyACM0')
    parser.add_argument('--serial_speed', metavar='number', required=False,
                        help='Serial port (ex: 57600)', type=int, default=57600)

    parser.add_argument('--record_video', metavar='boolean', required=False,
                        help='Record the video ', type=bool, default=False)
    parser.add_argument('--record_video_path', metavar='path', required=False,
                        help='Video destination path (ex: ./recordings/)', type=str, default='./')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')

    try:
        # Create serial manager
        communication_manager = SerialCommunicationManagerImpl(baud_rate=int(args.serial_speed),
                                                               serial_port=str(args.serial_port))
        # serial_manager.connect()

        # Create video tracker with serial manager
        tracked_bots = {
            72: UnleashTheBrickBot(name="tornado", tag_id=72, color=(255, 0, 0)),
            76: UnleashTheBrickBot(name="zigoto", tag_id=73, color=(0, 255, 0))}

        camera_number = args.camera_number
        if args.camera_number < 0:
            camera_number = AmakerBotTracker.user_input_camera_choice()
        reference_tags = {
            0: {"name": "north", type: "wall"},
            1: {"name": "east", type: "wall"},
            2: {"name": "south", type: "wall"},
            3: {"name": "west", type: "wall"},
            20: {"name": "north_east", type: "ground"},
            21: {"name": "north_west", type: "ground"},
            22: {"name": "south_east", type: "ground"},
            23: {"name": "south_west", type: "ground"},
        }
        vt = AmakerBotTracker(
            calibration_file=str(args.camera_calibration_file)
            , camera_index=int(args.camera_number)
            , communication_manager=communication_manager
            , tracked_bots=tracked_bots
            , window_size=(int(args.window_width), int(args.window_height))
            , know_tags=reference_tags)
        vt.start_tracking(recording=bool(args.record_video), recording_path=str(args.record_video_path))
    except Exception as e:
        logging.error(e)
        exit(1)
