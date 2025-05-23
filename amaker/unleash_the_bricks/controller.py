import datetime
import logging
import os
import cv2
import numpy as np

from amaker.communication.communication_abstract import CommunicationManagerAbstract
from amaker.communication.serial_communication_manager import SerialCommunicationManagerImpl
from amaker.detection.detector_apriltag import AprilTagDetectorImpl
from amaker.unleash_the_bricks import  \
    UI_RGBCOLOR_GREY_LIGHT, UI_RGBCOLOR_GREY_MEDIUM, DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT, UI_RGBCOLOR_GREY_DARK, \
    WINDOW_TITLE, UI_RGBCOLOR_ORANGE, UI_RGBCOLOR_HOTPINK, UI_RGBCOLOR_BRIGHTGREEN, UI_RGBCOLOR_LAVENDER, \
    VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT, VIDEO_OUT_FPS, VIDEO_OUT_CODEC, UI_RGBCOLOR_YELLOW
from amaker.unleash_the_bricks.bot import UnleashTheBrickBot
from gui_video import AmakerUnleashTheBrickVideo

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
CV_THREADS = 7
# Window settings
# CAMERA_SEARCH_LIMIT

CAMERA_SEARCH_LIMIT = 10

LOG_MAX_LINES=10

UI_COLOR_TAG_UNKNOWN = UI_RGBCOLOR_GREY_DARK
UI_COLOR_TAG_WALL = UI_RGBCOLOR_GREY_LIGHT
UI_COLOR_TAG_GROUND = UI_RGBCOLOR_GREY_MEDIUM
UI_COLOR_TAG_GOAL = UI_RGBCOLOR_YELLOW

# COLOR_TAG_IGNORED = UI_RGBCOLOR_GREY_MEDIUM
# COLOR_TAG_WALL = UI_RGBCOLOR_GREY_MEDIUM
# COLOR_TAG_GROUND = UI_RGBCOLOR_GREY_LIGHT
# COLOR_TAG_BOT = UI_RGBCOLOR_GREEN_LIGHT
# Video recording constants


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
    """
    Class to track the bots using a camera and AprilTag detection.
    This can be used standalone or embedded in a GUI.
    Problem with standalone : adding text on video can takes too much time and degrades the video FPS..
    """
    def __init__(self, calibration_file, camera_index: int = 0, tracked_bots: dict[int, UnleashTheBrickBot] = None,
                 communication_manager=None, window_size=(DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT),
                 know_tags: dict[int, dict] = None, max_logs=LOG_MAX_LINES, countdown_seconds:int=None):
        self._LOG = logging.getLogger(__name__)
        self.reference_tags = know_tags
        self.logs = []
        self.countdown_seconds = countdown_seconds
        self.max_logs = max_logs
        self.deadline:datetime=None
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
            self._LOG.info("Serial communication activated.")
        else:
            self._LOG.info("Serial communication not activated.")
        self.communication_manager.register_on_data_callback(self._on_data_received)
        self.bot_tracker = AprilTagDetectorImpl(calibration_file=self.calibration_file
                                                , detector_threads=CV_THREADS
                                                ,tag_size_cm=3
                                                ,)
        if isinstance(tracked_bots, dict):
            self.tracked_bots = tracked_bots
        else:
            self._LOG.error(
                "Initialization error : tracked_bots must be a dictionary of int->UnleashTheBrickBot instances")
        self.camera_index = self.user_input_camera_choice() if camera_index < 0 else camera_index

        self.video_capture = cv2.VideoCapture(self.camera_index)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_OUT_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_OUT_HEIGHT)

        cv2.setNumThreads(CV_THREADS)  # Set the number of threads for OpenCV

        if not self.video_capture.isOpened():
            raise ValueError(f"Camera {camera_index} not found or cannot be opened.")
        self.amaker_ui = AmakerUnleashTheBrickVideo(config={}
                                                    , buttons={
                "start": self._on_UI_BUTTON_start
                , "stop": self._on_UI_BUTTON_stop
                , "safety": self._on_UI_BUTTON_safety
            })

    def _cleanup_resources(self):
        try:
            if self.communication_manager:
                self.communication_manager.close()
                self._LOG.info("Communication closed.")
        except Exception as e:
            self._LOG.warning(f"Error during communication cleanup: {e}")

        try:
            self.video_capture.release()
            self._LOG.info("Video capture released.")
        except Exception as e:
            self._LOG.warning(f"Error during video capture release: {e}")

        try:
            if self.video_writer and self.video_writer.isOpened():
                # Release the video writer
                self.video_writer.release()
                self._LOG.info("Video writer released successfully.")
        except Exception as e:
            self._LOG.warning(f"Error during video capture release: {e}")

    # Button callback functions
    def _on_UI_BUTTON_start(self):
        """Handle start button click"""
        self.deadline=None
        if self.communication_manager:
            self.communication_manager.send(COMMAND_START)
            self._add_log(f"> {COMMAND_START} sent")
            self.deadline= None if self.countdown_seconds is None else datetime.datetime.now() + datetime.timedelta(seconds=self.countdown_seconds)
        else:
            self._add_log(f"! {COMMAND_START} failed to send")


    def _on_UI_BUTTON_stop(self):
        """Handle stop button click"""
        self.deadline=None
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
        #TODO: make the logic on message received.
        self._LOG.info(data)
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
    def overlay_countdown(self, frame):
        """
        Overlay countdown on the frame
        :param frame:
        :return:
        """
        if self.deadline:
            self.amaker_ui.ui_add_countdown(self.deadline)
    def overlay_tags(self, frame, tags):
        """
        Overlay detected tags on the frame
        :param frame:
        :param tags:
        :return:
        """
        for tag in tags:

            color = UI_COLOR_TAG_UNKNOWN
            #TODO : avoid duplicate ID in tracked_bots and in bot itself.tracked_bots
            if tag.tag_id in self.tracked_bots.keys():
                self.tracked_bots[tag.tag_id].add_tag_position(tag)
                self.amaker_ui.ui_add_bot(self.amaker_ui, self.mtx, self.dist, frame, self.tracked_bots[tag.tag_id])

            elif tag.tag_id in self.reference_tags.keys():
                if self.reference_tags[tag.tag_id]["type"] == "wall":
                    color = UI_COLOR_TAG_WALL
                elif self.reference_tags[tag.tag_id]["type"] == "ground":
                    color = UI_COLOR_TAG_GROUND
                elif self.reference_tags[tag.tag_id]["type"] == "goal":
                    color = UI_COLOR_TAG_GOAL
                self.amaker_ui.ui_add_tag(frame, self.mtx, self.dist, tag, color=color, label=self.reference_tags[tag.tag_id]["name"])
            else:
                self._LOG.debug(f"Tag {tag.tag_id} unknown : not shown. ")
            self._LOG.debug(f"BOT {tag.tag_id},  T={tag.pose_t} R={tag.pose_R} C={tag.center}")

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
                self._LOG.info(f"Video recording started: {video_name}")
            else:
                self._LOG.error(f"Video recording not started: {video_name}")
        else:
            self._LOG.info("No video recording.")

    def main_tracking_loop(self, window_name):
        while True:
            ret, input_frame = self.video_capture.read()
            #input_frame = cv2.resize(input_frame, self.window_size)
            if not ret:
                self._LOG.error("Failed to capture frame from camera.")
                break

            # Show the video feed
            detected_tags = self.bot_tracker.detect(input_frame)
            self.overlay_tags(input_frame, detected_tags)
            self.overlay_countdown(input_frame)
            self.amaker_ui.show_bot_infos(input_frame, self.tracked_bots)
            self.amaker_ui.show_logs(input_frame, self.logs)

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
        self._LOG.info(f"Bot trackers: {self.tracked_bots}")
        self._LOG.info(f"Press 'q' or ESC to quit.")
        try:
            self.setup_video_recording(recording, path=recording_path)
            self.main_tracking_loop(window_name)
        except KeyboardInterrupt as e:
            self._LOG.warning(f"Keyboard interruption.")
        except Exception as e:
            self._LOG.error(f"Error during video tracking: {e}")
            raise e
        finally:
            self._cleanup_resources()

    def __del__(self):
        """Destructor cleans up resources"""
        try:
            if self.communication_manager:
                self.communication_manager.close()
        except Exception as e:
            self._LOG.error(f"Error during serial cleanup: {e}")
        try:
            if hasattr(self, 'video_capture') and self.video_capture.isOpened():
                self.video_capture.release()
                self._LOG.info("Video capture released")
        except Exception as e:
            self._LOG.error(f"Error during capture cleanup: {e}")
        try:
            cv2.destroyAllWindows()
            self._LOG.info("All windows destroyed")
        except Exception as e:
            self._LOG.error(f"Error during winwow cleanup: {e}")


def main():

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
        tracked_bots_tagId = {
            71: UnleashTheBrickBot(name="team alfa", bot_id=1, rgb_color=UI_RGBCOLOR_ORANGE),
            72: UnleashTheBrickBot(name="team beta", bot_id=2, rgb_color=UI_RGBCOLOR_LAVENDER),
            73: UnleashTheBrickBot(name="team charly", bot_id=3, rgb_color=UI_RGBCOLOR_BRIGHTGREEN),
            74: UnleashTheBrickBot(name="team delta", bot_id=4, rgb_color=UI_RGBCOLOR_HOTPINK),
            }

        camera_number = args.camera_number
        if args.camera_number < 0:
            camera_number = AmakerBotTracker.user_input_camera_choice()
        reference_tags = {
            10: {"name": "north", type: "wall"},
            11: {"name": "east", type: "wall"},
            12: {"name": "south", type: "wall"},
            13: {"name": "west", type: "wall"},
            20: {"name": "north_east", type: "ground"},
            21: {"name": "north_west", type: "ground"},
            22: {"name": "south_east", type: "ground"},
            23: {"name": "south_west", type: "ground"},
            30: {"name": "goal", type: "goal"},
            31: {"name": "goal", type: "goal"},
            32: {"name": "goal", type: "goal"},
            33: {"name": "goal", type: "goal"},
        }
        vt = AmakerBotTracker(
            calibration_file=str(args.camera_calibration_file)
            , camera_index=int(args.camera_number)
            , communication_manager=communication_manager
            , tracked_bots=tracked_bots_tagId
            , window_size=(int(args.window_width), int(args.window_height))
            , know_tags=reference_tags)
        vt.start_tracking(recording=bool(args.record_video), recording_path=str(args.record_video_path))
    except Exception as e:
        logging.error(e)
        exit(1)


if __name__ == "__main__":
    main()