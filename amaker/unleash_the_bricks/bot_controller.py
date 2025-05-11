import datetime
import logging
import os
from typing import List

import cv2
import numpy as np
from pyapriltags import Detector, Detection

from amaker.unleash_the_bricks.bot import UnleashTheBrickBot
# Import SerialManager from the new file
from amaker.serial_communication.serial_manager import SerialManager
from unleash_the_brick_ui import AmakerUnleashTheBrickUI

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
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080

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

COMMAND_START = "123456789-12345679-123456789-123456789-"
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


class AmakerApriltagTracker():
    """
    Class to track AprilTags using OpenCV and pyapriltags.
    """
    def __init__(self, calibration_file: str = 'camera_calibration.npz', detector_threads: int = 4,
                 tag_family: str = 'tag36h11'):
        self.apriltag_detector = Detector(families=tag_family, nthreads=detector_threads, quad_sigma=0.0,
                                          refine_edges=1, decode_sharpening=0.25, debug=0)

        calibration_data = np.load(calibration_file)
        self.mtx = calibration_data['camera_matrix']
        self.dist = calibration_data['dist_coeffs']
        self.fx = self.mtx[0, 0]
        self.fy = self.mtx[1, 1]  # Focal length in y direction
        self.cx = self.mtx[0, 2]  # Principal point x-coordinate (optical center)
        self.cy = self.mtx[1, 2]  # Principal point y-coordinate (optical center)
        self.camera_params = [self.fx, self.fy, self.cx, self.cy]
        self.video_writer = None

    def detect(self, frame, tag_size_cm=10) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_undistorted = cv2.undistort(gray, self.mtx, self.dist, None, newCameraMatrix=self.mtx)
        detected = self.apriltag_detector.detect(gray_undistorted, estimate_tag_pose=True,
                                                 camera_params=self.camera_params, tag_size=tag_size_cm/100
                                                 )
        return detected


class AmakerBotTracker():
    def __init__(self, calibration_file, camera_index: int = 0, tracked_bots: List[UnleashTheBrickBot] = None, serial_manager=None, window_size=(640,480)):

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



        self.bot_tracker = AmakerApriltagTracker(calibration_file= self.calibration_file  )
        self.tracked_bots =tracked_bots
        self.camera_index = self.user_input_camera_choice() if camera_index < 0 else camera_index

        self.video_capture = cv2.VideoCapture(self.camera_index)
        cv2.setNumThreads(CV_THREADS)  # Set the number of threads for OpenCV
        cv2.setLogLevel(2)  # Set OpenCV log severity to no logs
        if not self.video_capture.isOpened():
            raise ValueError(f"Camera {camera_index} not found or cannot be opened.")
        self.amaker_ui = AmakerUnleashTheBrickUI({}
                                                 , self._on_UI_BUTTON_start
                                                 , self._on_UI_BUTTON_stop
                                                 , self._on_UI_BUTTON_safety())

    # Button callback functions
    def _on_UI_BUTTON_start(self):
        """Handle start button click"""

        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_START)
            self._add_log(f"sent: {COMMAND_START}")
        else:
            self._add_log(f"unsent: {COMMAND_START}")

    def _on_UI_BUTTON_stop(self):
        """Handle stop button click"""
        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_STOP)
            self._add_log(f"sent: {COMMAND_STOP}")
        else:
            self._add_log(f"unsent: {COMMAND_STOP}")

    def _on_UI_BUTTON_safety(self):
        """Handle safety button click"""
        if self.serial_manager:
            self.serial_manager.send_command(COMMAND_SAFETY)
            self._add_log(f"sent: {COMMAND_SAFETY}")
        else:
            self._add_log(f"unsent: {COMMAND_SAFETY}")

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

    def user_input_camera_choice(self) -> int | None:
        """Select a camera from available cameras"""
        index = 0
        available_cameras = []
        logging.info(f"Searching for cameras...")
        cap = None
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

    # def process_bot_tracking(self, frame):
    #     for bot in self.tracked_bots:
    #         try:
    #             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             bot_mask = cv2.inRange(rgb, bot.color_a, bot.color_b)
    #             bot_contours, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #             if len(bot_contours) > 0:
    #                 self._update_bot_position(bot, bot_contours, frame)
    #
    #             else:
    #                 logging.debug(f"No contours found for {bot.name}")
    #         except Exception as e:
    #             logging.error(f"Error tracking {bot.name}: {e}")

    def setup_video_recording(self, recording: bool = False):
        """Setup video recording"""
        if recording:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f'aMaker_microbot_tracker_{current_time}.avi'
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
                        rvec, _ = cv2.Rodrigues(tag.pose_R)  # Convert rotation matrix to rotation vector
                        tvec = pose_t

                        # Project the 3D points directly with proper rvec and tvec
                        image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.mtx, self.dist)
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

                    logging.info(f"BOT {tag.tag_id},  T={tag.pose_t} R={tag.pose_R} C={tag.center}")


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
                cv2.putText(frame, log, (10, y_pos), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 1)

    def main_tracking_loop(self, window_name, video_writer):
        while True:
            ret, input_frame = self.video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break

            # Convert to HSV for color detection
            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGRA2RGB)

            # Draw detected tags

            # TODO HERE the video analysis
            # for bot_tracker in self.tracked_bots:
            #     # Get the bot's color range
            #     bot_mask = cv2.inRange(rgb, bot_tracker.color_a, bot_tracker.color_b)
            #     bot_contours, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            #     # Extract bot positions
            #     if bot_contours:
            #         x, y, w, h = cv2.boundingRect(bot_contours[0])
            #         bot_position = (x + w // 2, y + h // 2)
            #         bot_tracker.add_position(bot_position)
            #
            #         if len(bot_tracker.get_trail()) > 1:
            #             cv2.polylines(input_frame, [np.array(bot_tracker.get_trail())], False, bot_tracker.trail_color,
            #                           2)
            #             cv2.circle(input_frame, bot_position, 5, bot_tracker.trail_color, -1)

            # Add text to the frame

            if video_writer:
                # Save the frame to the video file
                video_writer.write(input_frame)

            # Show the video feed
            detected_tags = self.bot_tracker.detect(input_frame)
            self.overlay_detected_tags(input_frame, detected_tags)
            scaled_frame = cv2.resize(input_frame, self.window_size)
            display_frame = self.amaker_ui.build_display_frame(scaled_frame)
            self.overlay_bot_infos(display_frame)
            self.overlay_logs(display_frame)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_Q or key == KEY_CTRL_D or key == KEY_ESC:  # 'q' or Ctrl+D or ESC to quit
                break

    def overlay_bot_infos(self, frame):
        # Print bots infos
        for i, bot_tracker in enumerate(self.tracked_bots):
            y_pos = frame.shape[0] - 30 - (i * 20)
            cv2.putText(frame, bot_tracker.get_bot_info(), (1090, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        bot_tracker.trail_color, 2)

    def start_tracking(self, recording: bool = False, window_name=WINDOW_TITLE):
        """Start tracking bots in video feed"""
        input_frame, original_height, original_width = self.amaker_ui.initialize_window(self.video_capture, window_name,on_start=self._on_UI_BUTTON_start,on_stop=self._on_UI_BUTTON_stop,on_safety=self._on_UI_BUTTON_safety)
        logging.info(f"Bot trackers: {self.tracked_bots}")
        logging.info(f"Press 'q' or ESC to quit.")
        logging.info(f"Press 'f' to toggle full screen.")
        try:
            self.video_writer = self.setup_video_recording(recording)
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Create serial manager
    serial_manager = SerialManager()
    serial_manager.connect_serial(baud_rate=57600, port="/dev/ttyACM0")

    # Create video tracker with serial manager
    tracked_bots = [
        UnleashTheBrickBot(name="bot72", tag_id=72, trail_color=(0, 255, 0), trail_length=10) ]
    vt = AmakerBotTracker(
        calibration_file="/home/taccart/VSCode/amaker-path-capture/camera_calibration.npz"
        , camera_index=4
        , serial_manager=serial_manager
        , tracked_bots=tracked_bots
        , window_size=(1920, 1080))
    vt.start_tracking(recording=False)
