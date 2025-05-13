import datetime
import logging
import os
from typing import List

import cv2
import numpy as np

# Import SerialManager from the new file
from amaker.communication.serial_communication_manager import SerialCommunicationManagerImpl
from amaker.detection.detector_apriltag import AprilTagDetectorImpl
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
CV_THREADS = 4
# Window settings
# CAMERA_SEARCH_LIMIT

CAMERA_SEARCH_LIMIT = 10
WINDOW_TITLE = "aMaker microbot tracker"
DEFAULT_SCREEN_WIDTH = 1280
DEFAULT_SCREEN_HEIGHT = 720
UI_BOT_INFO_FONT_THICKNESS = 2
UI_BOT_INFO_FONT_COLOR = (255, 0, 0)
UI_BOT_INFO_FONT_SCALE = 0.6
UI_BOT_INFO_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_BOT_INFO_X = 10
UI_BOT_INFO_Y = 60
UI_BOT_INFO_Y_DELTA = 25
UI_LOG_FONT_COLOR = (128, 128, 128)
UI_LOG_FONT_THICKNESS = 1
UI_LOG_FONT_SCALE = 0.8
COLOR_IGNORED = (255, 255, 255)
COLOR_WALL = (255, 255, 100)
COLOR_GROUND = (200, 255, 100)
COLOR_BOT = (100, 100, 255)
# Video recording constants
VIDEO_CODEC = 'XVID'
VIDEO_FPS = 30.0
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
APRILTAG_SIZE = 0.1  # Size of the AprilTag in meters
# Key codes
KEY_ESC = 27
KEY_CTRL_D = 4
KEY_Q = ord('q')
KEY_F = ord('f')

COMMAND_START = "START"
COMMAND_STOP = "STOP"
COMMAND_SAFETY = "SAFETY"

# Bot tracker constants


reference_tags = {
    "wall": {
        0: {"name": "north", },
        1: {"name": "east", },
        2: {"name": "south", },
        3: {"name": "west", },
        # 4:{"name":"unused_4", },
        # 5:{"name":"unused_5", },
        # 6:{"name":"unused_6", },
        # 7:{"name":"unused_7", },
        # 8:{"name":"unused_8", },
        # 9:{"name":"unused_9", },
    },
    "ground": {
        20: {"name": "north_east", },
        21: {"name": "north_west", },
        22: {"name": "south_east", },
        23: {"name": "south_west", },
        # 24:{"name":"ground_unused_24", },
        # 25:{"name":"ground_unused_25", },
        # 26:{"name":"ground_unused_26", },
        # 27:{"name":"ground_unused_27", },
        # 28:{"name":"ground_unused_28", },
        # 29:{"name":"ground_unused_29", },
    },
    "bot": {
        70: {"name": "bot (id 70)"},
        71: {"name": "bot (id 71)"},
        72: {"name": "bot (id 72)"},
        73: {"name": "bot (id 73)"},
        74: {"name": "bot (id 74)"},
        75: {"name": "bot (id 75)"},
        76: {"name": "bot (id 76)"},
        77: {"name": "bot (id 77)"},
        78: {"name": "bot (id 78)"},
        79: {"name": "bot (id 79)"},
    }
}


class AmakerBotTracker():
    def __init__(self, calibration_file, camera_index: int = 0, tracked_bots: List[UnleashTheBrickBot] = None,
                 serial_manager=None, window_size=(640, 480)):

        self.logs = []
        self.max_logs = 5
        self.window_size = window_size
        self.calibration_file = calibration_file
        calibration_data = np.load(calibration_file)
        self.mtx = calibration_data['camera_matrix']
        self.dist = calibration_data['dist_coeffs']
        self.serial_manager = serial_manager
        if serial_manager is not None:
            logging.info("Serial communication activated.")
        else:
            logging.info("Serial communication not activated.")

        self.bot_tracker = AprilTagDetectorImpl(calibration_file=self.calibration_file)
        self.tracked_bots = tracked_bots
        self.camera_index = self.user_input_camera_choice() if camera_index < 0 else camera_index

        self.video_capture = cv2.VideoCapture(self.camera_index)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.setNumThreads(CV_THREADS)  # Set the number of threads for OpenCV
        if not self.video_capture.isOpened():
            raise ValueError(f"Camera {camera_index} not found or cannot be opened.")
        self.amaker_ui = AmakerUnleashTheBrickGUI(config={}
                                                  , buttons={"start" : self._on_UI_BUTTON_start
                                                  , "stop":self._on_UI_BUTTON_stop
                                                  , "safety":self._on_UI_BUTTON_safety})

    # Button callback functions
    def _on_UI_BUTTON_start(self):
        """Handle start button click"""

        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_START)
            self._add_log(f"> {COMMAND_START} sent")
        else:
            self._add_log(f"! {COMMAND_START} failed to send")

    def _on_UI_BUTTON_stop(self):
        """Handle stop button click"""
        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_STOP)
            self._add_log(f"> {COMMAND_STOP}")
        else:
            self._add_log(f"! {COMMAND_STOP} failed to send")

    def _on_UI_BUTTON_safety(self):
        """Handle safety button click"""
        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_SAFETY)
            self._add_log(f"> {COMMAND_SAFETY}")
        else:
            self._add_log(f"! {COMMAND_SAFETY} failed to send")

    def _add_log(self, message):
        current_time = datetime.datetime.now().strftime("%H%M%S")
        self.logs.append(current_time + " " + str(message))
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def bot_assign_colors(self):
        ## broadcast new colors of all bots
        raise NotImplementedError

    def bot_verify_colors(self):
        ## check aknowledgement of new colors for all bots
        raise NotImplementedError

    def _update_bot_position(self):
        raise NotImplementedError

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

    def setup_video_recording(self, is_recording: bool, path: str) -> cv2.VideoWriter:
        """Setup video recording"""
        if is_recording:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f'{path}aMaker_microbot_tracker_{current_time}.avi'
            self.video_writer = cv2.VideoWriter(video_name, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
            logging.info(f"Video recording started: {video_name}")
        else:
            logging.info("No video recording.")

    def _cleanup_resources(self):
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
            if self.video_writer:
                # Release the video writer
                self.video_writer.release()
                logging.info("Video writer released successfully.")
        except Exception as e:
            logging.warning(f"Error during video capture release: {e}")

    def overlay_detected_tags(self, frame, tags):
        for tag in tags:
            if tag.tag_id in reference_tags["wall"]:
                for idx in range(len(tag.corners)):
                    cv2.line(frame, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                             COLOR_GROUND, 5)
                    cv2.putText(frame, reference_tags["wall"][tag.tag_id]["name"],
                                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_GROUND, thickness=1)

            elif tag.tag_id in reference_tags["ground"]:
                for idx in range(len(tag.corners)):
                    cv2.line(frame, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                             COLOR_GROUND, 5)
                    cv2.putText(frame, reference_tags["ground"][tag.tag_id]["name"],
                                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_GROUND, thickness=1)

            elif tag.tag_id in reference_tags["bot"]:

                for idx in range(len(tag.corners)):
                    # cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)),tuple(tag.corners[idx, :].astype(int)), COLOR_BOT,5)
                    cv2.putText(frame, reference_tags["bot"][tag.tag_id]["name"],
                                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) - 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_BOT, thickness=1)

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
                        cv2.line(frame, origin, tuple(image_points[1]), (0, 0, 255), 3)  # X-axis (red)
                        cv2.line(frame, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y-axis (green)
                        cv2.line(frame, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z-axis (blue)

                        # Add labels to the axes for better visibility
                        cv2.putText(frame, "X", tuple(image_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, "Y", tuple(image_points[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, "Z", tuple(image_points[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Log axis points for debugging
                        # logging.debug(f"Axis points: Origin={image_points[0]}, X={image_points[1]}, Y={image_points[2]}, Z={image_points[3]}")
                    else:
                        logging.error(f"Tag {tag.tag_id} missing pose information")

                    logging.debug(f"BOT {tag.tag_id},  T={tag.pose_t} R={tag.pose_R} C={tag.center}")


            else:
                for idx in range(len(tag.corners)):
                    cv2.putText(frame
                                , str(tag.tag_id)
                                , org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10)
                                , fontFace=cv2.FONT_HERSHEY_SIMPLEX
                                , fontScale=1
                                , color=COLOR_IGNORED
                                , thickness=1)
                    cv2.line(frame
                             , tuple(tag.corners[idx - 1, :].astype(int))
                             , tuple(tag.corners[idx, :].astype(int))
                             , color=COLOR_IGNORED
                             , thickness=5)

    def overlay_logs(self, frame):
        if self.logs:
            for i, log in enumerate(reversed(self.logs)):
                y_pos = frame.shape[0] - 30 - (i * 20)
                cv2.putText(frame, log, (10, y_pos), cv2.FONT_HERSHEY_PLAIN, UI_LOG_FONT_SCALE, UI_LOG_FONT_COLOR,
                            UI_LOG_FONT_THICKNESS)


    def overlay_bot_infos(self, frame):
        # Print bots infos
        for i, bot_tracker in enumerate(self.tracked_bots):
            y_pos =  UI_BOT_INFO_Y +  (i * UI_BOT_INFO_Y_DELTA)
            cv2.putText(frame, bot_tracker.get_bot_info()
                        , (UI_BOT_INFO_X, y_pos)
                        , UI_BOT_INFO_FONT, UI_BOT_INFO_FONT_SCALE,
                        UI_BOT_INFO_FONT_COLOR, UI_BOT_INFO_FONT_THICKNESS)


    def main_tracking_loop(self, window_name, video_writer):
        while True:
            ret, input_frame = self.video_capture.read()
            input_frame = cv2.resize(input_frame, self.window_size)
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break

            # Convert to HSV for color detection
            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGRA2RGB)

            if video_writer:
                # Save the frame to the video file
                video_writer.write(input_frame)

            # Show the video feed
            detected_tags = self.bot_tracker.detect(input_frame)
            self.overlay_detected_tags(input_frame, detected_tags)

            display_frame = self.amaker_ui.build_display_frame(input_frame)
            self.overlay_bot_infos(display_frame)
            self.overlay_logs(display_frame)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_Q or key == KEY_CTRL_D or key == KEY_ESC:  # 'q' or Ctrl+D or ESC to quit
                break

    def start_tracking(self, recording: bool = False, recording_path: str = ".", window_name=WINDOW_TITLE):
        """Start tracking bots in video feed"""
        input_frame, original_height, original_width = self.amaker_ui.initialize_window(self.video_capture, window_name,
                                                                                        on_start=self._on_UI_BUTTON_start,
                                                                                        on_stop=self._on_UI_BUTTON_stop,
                                                                                        on_safety=self._on_UI_BUTTON_safety)
        logging.info(f"Bot trackers: {self.tracked_bots}")
        logging.info(f"Press 'q' or ESC to quit.")
        try:
            self.video_writer = self.setup_video_recording(recording, path=recording_path)
            self.main_tracking_loop(window_name, self.video_writer)
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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Create serial manager
        serial_manager = SerialCommunicationManagerImpl(baud_rate=int(args.serial_speed),
                                                        serial_port=str(args.serial_port))
        # serial_manager.connect()

        # Create video tracker with serial manager
        tracked_bots = [
            UnleashTheBrickBot(name="static bot", tag_id=72, trail_color=(0, 255, 0), trail_length=10),
            UnleashTheBrickBot(name="moving bot ", tag_id=73, trail_color=(0, 255, 0), trail_length=10)]
        camera_number = args.camera_number
        if args.camera_number < 0:
            camera_number = AmakerBotTracker.user_input_camera_choice()

        vt = AmakerBotTracker(
            calibration_file=str(args.camera_calibration_file)
            , camera_index=int(args.camera_number)
            , serial_manager=serial_manager
            , tracked_bots=tracked_bots
            , window_size=(int(args.window_width), int(args.window_height)))
        vt.start_tracking(recording=bool(args.record_video), recording_path=str(args.record_video_path))

    except Exception as e:
        logging.error(e)
        exit(1)
