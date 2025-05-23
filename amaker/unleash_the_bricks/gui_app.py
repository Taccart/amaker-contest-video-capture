import time
import cv2
import sys
import argparse
import logging
import os
import yaml
from typing import Dict, Any

# PyQt imports - grouped together and explicitly formatted
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout,
     QWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QDockWidget, QMenuBar
)
from PyQt6.QtCore import Qt, QTimer, QSettings

from PyQt6.QtGui import QPixmap, QImage, QAction
from amaker.unleash_the_bricks.controller import AmakerBotTracker
from amaker.unleash_the_bricks.bot import UnleashTheBrickBot
from amaker.communication.serial_communication_manager import SerialCommunicationManagerImpl
from amaker.unleash_the_bricks import DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT, VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT, \
    UI_RGBCOLOR_BRIGHTGREEN

# Set environment variable for Qt
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Constants
VIDEO_REFRESH_FPS = 30

DEFAULT_WINDOW_MAIN_X=100
DEFAULT_WINDOW_MAIN_Y=100
DEFAULT_WINDOW_MAIN_WIDHT=1280
DEFAULT_WINDOW_MAIN_HEIGHT=1024

DEFAULT_WINDOW_LOG_X=10
DEFAULT_WINDOW_LOG_Y=50
DEFAULT_WINDOW_LOG_WIDHT=500
DEFAULT_WINDOW_LOG_HEIGHT=400

DEFAULT_WINDOW_BOTS_X=60
DEFAULT_WINDOW_BOTS_Y=10
DEFAULT_WINDOW_BOTS_WIDHT=1024
DEFAULT_WINDOW_BOTS_HEIGHT=100
DEFAULT_WINDOW_BOTS_MAX_SHOWN=6

REFRESH_LOG_MS =200
REFRESH_TABLE_MS = 200

VIDEO_DEFAULT_BOT_COLOR = UI_RGBCOLOR_BRIGHTGREEN

class LogWindow(QDockWidget):
    def __init__(self):
        super().__init__("Logs")
        self._LOG = logging.getLogger(__name__)
        self.setGeometry(DEFAULT_WINDOW_LOG_X, DEFAULT_WINDOW_LOG_Y, DEFAULT_WINDOW_LOG_WIDHT, DEFAULT_WINDOW_LOG_HEIGHT)

        # Create a widget to hold the text edit
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        self.logs_text = QTextEdit()
        self.logs_text.setFontFamily("Courier")
        self.logs_text.setFontPointSize(8)
        self.logs_text.setReadOnly(True)

        layout.addWidget(self.logs_text)
        content_widget.setLayout(layout)

        # Set the widget as the dockwidget's content
        self.setWidget(content_widget)

        # Make the dock widget floatable and closable
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable| QDockWidget.DockWidgetFeature.DockWidgetMovable )

    def append_log(self, message):
        self.logs_text.append(message)

    def update_text(self, logs: list[str]|None):
        if logs is None:
            self.logs_text.setText("")
        elif len(logs) > 0:
            self.logs_text.setText("\n".join(logs))
        else:
            self.logs_text.setText("")

class BotsInfoWindow(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bots")
        self.setGeometry(DEFAULT_WINDOW_BOTS_X, DEFAULT_WINDOW_BOTS_Y, DEFAULT_WINDOW_BOTS_WIDHT, DEFAULT_WINDOW_BOTS_HEIGHT)
        # Initialize with 0 rows, will resize dynamically
        self.table_widget = QTableWidget(0, 4)
        self.table_widget.setHorizontalHeaderLabels(["Team", "bot state", "distance", "self est. count"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setWidget(self.table_widget)

        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable| QDockWidget.DockWidgetFeature.DockWidgetMovable )


    def update_table(self, bots:list[UnleashTheBrickBot]):
        current_row_count = self.table_widget.rowCount()
        bots_count = len(bots)

        if current_row_count != bots_count:
            self.table_widget.setRowCount(bots_count)

        for row, bot in enumerate(bots):
            self.table_widget.setItem(row, 0, QTableWidgetItem(bot.name))
            self.table_widget.setItem(row, 1, QTableWidgetItem(str(bot.status)))
            self.table_widget.setItem(row, 2, QTableWidgetItem(f"{bot.total_distance/100:6.2f}"))
            self.table_widget.setItem(row, 3, QTableWidgetItem(str(bot.collected_count)))

        # Resize window to fit table contents
        self.resize_window_to_fit_table()

    def resize_window_to_fit_table(self):
        # Get header height
        header_height = self.table_widget.horizontalHeader().height()

        # Calculate total row heights
        row_height = 30  # Default row height
        total_row_height = 0
        for i in range(self.table_widget.rowCount()):
            total_row_height += self.table_widget.rowHeight(i) or row_height

        # Add some padding
        padding = 40

        # Calculate new height and set it (keeping current width)
        new_height = header_height + total_row_height + padding
        current_width = self.width()

        # Set minimum height for empty tables
        min_height = 100
        new_height = max(new_height, min_height)

        # Update window size
        self.resize(current_width, new_height)

class AmakerControllerUI(QMainWindow):
    def __init__(self, controller: AmakerBotTracker, max_tracked_bots_count=4):
        super().__init__()
        self._LOG = logging.getLogger(__name__)
        self.setWindowTitle("Unleash The Bricks")
        self.setGeometry(DEFAULT_WINDOW_MAIN_X, DEFAULT_WINDOW_MAIN_Y, DEFAULT_WINDOW_MAIN_WIDHT, DEFAULT_WINDOW_MAIN_HEIGHT)
        self.max_tracked_bots_count = max_tracked_bots_count
        self.controller = controller

        # Restore window state if available
        self.settings = QSettings("AmakerBot", "UnleashTheBricks")
        if self.settings.contains("mainWindowGeometry"):
            self.restoreGeometry(self.settings.value("mainWindowGeometry"))
        if self.settings.contains("mainWindowState"):
            self.restoreState(self.settings.value("mainWindowState"))

        # Create menu bar : Order and View
        self.menu = self.menuBar()
        self.setup_menu_bar(self.menu)


        # Create and add dock widgets
        self.bots_info_window = BotsInfoWindow()
        self.bots_info_window.setWindowTitle("Bots")
        self.bots_info_window.setObjectName("bots_info_window")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.bots_info_window)
        self.bots_info_window.visibilityChanged.connect(self.toggle_bot_info_action)

        # Create communication logs window (if it doesn't exist)
        self.comm_logs_window = LogWindow()
        self.comm_logs_window.setWindowTitle("Communication logs")
        self.comm_logs_window.setObjectName("communication_logs_window")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.comm_logs_window)
        self.comm_logs_window.visibilityChanged.connect(self.toggle_communication_logs)

        self.system_logs_window = LogWindow()
        self.system_logs_window.setWindowTitle("System logs")
        self.system_logs_window.setObjectName("system_logs_window")
          # Hidden by default
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.system_logs_window)
        self.system_logs_window.visibilityChanged.connect(self.toggle_system_logs_action)


        main_layout = QVBoxLayout()

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        #
        # # Title and buttons
        # title_layout = QHBoxLayout()
        # title_label = QLabel("Unleash The Bricks")
        # title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # title_layout.addWidget(title_label)
        # start_button = QPushButton("Start")
        # stop_button = QPushButton("Stop")
        # safety_button = QPushButton("Safety")
        # title_layout.addWidget(start_button)
        # title_layout.addWidget(stop_button)
        # title_layout.addWidget(safety_button)
        # main_layout.addLayout(title_layout)
        # Button actions
        # start_button.clicked.connect(self.start_button_clicked)
        # stop_button.clicked.connect(self.stop_button_clicked)
        # safety_button.clicked.connect(self.safety_button_clicked)

        # Video stream
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.video_label)

        # Video timer
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_frame)

        # Table update timer
        self.table_timer = QTimer()
        self.table_timer.timeout.connect(self.update_table)
        self.table_timer.start(REFRESH_TABLE_MS)  # Update table every 500ms

        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_log)
        self.log_timer.start(REFRESH_LOG_MS)  # Update table every 500ms

    def setup_menu_bar(self, menuBar:QMenuBar):

        self.order_menu = menuBar.addMenu("Order")
        self.view_menu = menuBar.addMenu("View")


        # Add menu actions for dock widgets
        self.system_logs_action = QAction("System Logs", self)
        self.system_logs_action.setCheckable(True)
        self.system_logs_action.setChecked(True)
        self.system_logs_action.triggered.connect(self.toggle_system_logs_window)

        self.bots_info_action = QAction("Bots table", self)
        self.bots_info_action.setCheckable(True)
        self.bots_info_action.setChecked(True)
        self.bots_info_action.triggered.connect(self.toggle_bots_info_window)

        self.comm_logs_action = QAction("Communication Logs", self)
        self.comm_logs_action.setCheckable(True)
        self.comm_logs_action.setChecked(False)
        self.comm_logs_action.triggered.connect(self.toggle_comm_logs_window)

        # Add actions to view menu
        self.view_menu.addAction(self.system_logs_action)
        self.view_menu.addAction(self.bots_info_action)
        self.view_menu.addAction(self.comm_logs_action)


        # Create menu bar (after existing menu bar code)

        # Add menu actions for order controls
        self.start_action = QAction("Start", self)
        self.start_action.triggered.connect(self.start_button_clicked)
        self.order_menu.addAction(self.start_action)

        self.stop_action = QAction("Stop", self)
        self.stop_action.triggered.connect(self.stop_button_clicked)
        self.order_menu.addAction(self.stop_action)

        self.safety_action = QAction("Safety", self)
        self.safety_action.triggered.connect(self.safety_button_clicked)
        self.order_menu.addAction(self.safety_action)


    def toggle_system_logs_window(self):
        if self.system_logs_window.isVisible():
            self.system_logs_window.hide()
        else:
            self.system_logs_window.show()

    def toggle_bots_info_window(self):
        if self.bots_info_window.isVisible():
            self.bots_info_window.hide()
        else:
            self.bots_info_window.show()

    def toggle_comm_logs_window(self):
        if self.comm_logs_window.isVisible():
            self.comm_logs_window.hide()
        else:
            self.comm_logs_window.show()

    def toggle_bot_info_action(self, visible):
        self.bots_info_action.setChecked(visible)

    def toggle_system_logs_action(self, visible):
        self.system_logs_action.setChecked(visible)

    def toggle_communication_logs(self,  visible:bool):
        self.comm_logs_action.setChecked(visible)

    def update_log(self):
        if self.controller:
            self.comm_logs_window.update_text(self.controller.logs)

    def update_frame(self):
        if self.controller:
            ret, frame = self.controller.video_capture.read()
            if ret:
                # Process the frame using the controller
                detected_tags = self.controller.bot_tracker.detect(frame)
                self.controller.overlay_tags(frame, detected_tags)
                self.controller.amaker_ui.show_bot_infos(frame, self.controller.tracked_bots)

                self.controller.amaker_ui.ui_add_countdown(frame, self.controller.deadline)


                # Convert the frame to QImage and display it
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                q_img = QImage(frame.data, width, height, step, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
                if self.controller.video_writer and self.controller.video_writer.isOpened():
                    # Save the frame to the video file
                    video_out = cv2.resize(frame, (VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT))
                    self.controller.video_writer.write(video_out)

    def update_table(self):
        if self.controller and hasattr(self.controller, 'tracked_bots'):
            bots = list(self.controller.tracked_bots.values())
            self.bots_info_window.update_table(bots)


    def start_button_clicked(self):
        if self.controller:
            self.controller._on_UI_BUTTON_start()
        else:
            self._LOG.error("Controller is not initialized. Cannot handle start.")


    def stop_button_clicked(self):
        if self.controller:
            self.controller._on_UI_BUTTON_stop()
        else:
            self._LOG.error("Controller is not initialized. Cannot handle stop.")


    def safety_button_clicked(self):
        if self.controller:
            self.controller._on_UI_BUTTON_safety()
        else:
            self._LOG.error("Controller is not initialized. Cannot handle safety.")


    def closeEvent(self, event):
        if self.controller:
            self.controller._cleanup_resources()
        # Save the dock widget positions and sizes
        self.settings = QSettings("AmakerBot", "UnleashTheBricks")
        self.settings.setValue("mainWindowGeometry", self.saveGeometry())
        self.settings.setValue("mainWindowState", self.saveState())
        super().closeEvent(event)

def parse_arguments():
    """Parse command line arguments - now only config file path"""
    parser = argparse.ArgumentParser(description='Unleash the bricks: GUI application')
    parser.add_argument('--config', metavar='path', required=True,
                        help='Path to YAML configuration file', type=str)
    return parser.parse_args()

def load_and_validate_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file and apply defaults/validation"""
    try:
        # Load config from file
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Define default values
        defaults = {
            "window": {
                "width": DEFAULT_SCREEN_WIDTH,
                "height": DEFAULT_SCREEN_HEIGHT
            },
            "camera": {
                "calibration_file": "camera_calibration.npz",
                "number": -1
            },
            "communication": {
                "serial_port": "/dev/ttyACM0",
                "serial_speed": 57600
            },
            "recording": {
                "enabled": False,
                "path": "./"
            },
            "logs": {
                "max_count": 10,
                "show_on_screen": False
            }
        }
        if "logs" not in config:
            config["logs"] = {}
        config["logs"]["max_count"] = config["logs"].get("max_count", defaults["logs"]["max_count"])
        config["logs"]["show_on_screen"] = config["logs"].get("show_on_screen", defaults["logs"]["show_on_screen"])

    # Check for mandatory settings
        if not config.get("camera", {}).get("calibration_file"):
            raise ValueError("Missing required setting: camera.calibration_file")

        # Apply defaults for missing values
        if "window" not in config:
            config["window"] = {}
        config["window"]["width"] = config["window"].get("width", defaults["window"]["width"])
        config["window"]["height"] = config["window"].get("height", defaults["window"]["height"])

        if "camera" not in config:
            config["camera"] = {}
        config["camera"]["number"] = config["camera"].get("number", defaults["camera"]["number"])

        if "communication" not in config:
            config["communication"] = {}
        config["communication"]["serial_port"] = config["communication"].get("serial_port",
                                                                             defaults["communication"]["serial_port"])
        config["communication"]["serial_speed"] = config["communication"].get("serial_speed",
                                                                              defaults["communication"]["serial_speed"])

        if "recording" not in config:
            config["recording"] = {}
        else:
            if "path" not in  config["recording"]:
                config["recording"]["path"] = defaults["recording"]["path"]

        config["recording"]["enabled"] = config["recording"].get("enabled",
                                                                 defaults["recording"]["enabled"])
        config["recording"]["path"] = config["recording"].get("path", defaults["recording"]["path"])

        return config

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {config_file}")
        raise (e)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise (e)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')

    try:
        # Parse command line arguments to get config file
        args = parse_arguments()

        # Load and validate configuration
        config = load_and_validate_config(args.config)

        # Initialize the application
        app = QApplication(sys.argv)



        # Process known tags
        tracked_bots = {}
        reference_tags = {}

        for tag_id_str, tag_info in config["known_tags"].items():
            tag_id = int(tag_id_str)
            tag_type = tag_info.get("type", "")

            if tag_type == "bot":
                tracked_bots[tag_id] = UnleashTheBrickBot(
                    name=tag_info.get("name", f"Bot {tag_id}"),
                    bot_id=tag_id,
                    rgb_color=tuple(tag_info.get("color", VIDEO_DEFAULT_BOT_COLOR)) if tag_info.get("color") else VIDEO_DEFAULT_BOT_COLOR,                )
            else:
                reference_tags[tag_id] = {
                    "name": tag_info.get("name", ""),
                    "type": tag_type
                }

        # Select camera
        camera_number = config["camera"]["number"]
        if camera_number < 0:
            camera_number = AmakerBotTracker.user_input_camera_choice()
        # Create communication manager
        communication_manager = SerialCommunicationManagerImpl(
            baud_rate=config["communication"]["serial_speed"],
            serial_port=config["communication"]["serial_port"]
        )
        # Initialize the bot tracker
        controller = AmakerBotTracker(
            calibration_file=config["camera"]["calibration_file"],
            camera_index=camera_number,
            communication_manager=communication_manager,
            tracked_bots=tracked_bots,
            window_size=(config["window"]["width"], config["window"]["height"]),
            know_tags=reference_tags,
            max_logs=config["logs"]["max_count"],
            countdown_seconds=config["countdown_second"]

        )



        window = AmakerControllerUI(controller=controller)


        window.controller.amaker_ui.is_overlay_logs = config["logs"]["show_on_screen"]


        # Start video timer
        window.video_timer.start(VIDEO_REFRESH_FPS)
        controller.communication_manager.start_reading()
        time.sleep(0.2)

        # Show the window
        window.show()

        # Configure video recording if enabled
        if config["recording"]["enabled"]:
            if config["recording"]["path"]:
                if os.path.isdir(config["recording"]["path"]):
                    window.controller.setup_video_recording(True, config["recording"]["path"])
                else:
                    logging.error(f"NO RECORDING  - Invalid recording path: {config['recording']['path']}")

        # Run the application
        sys.exit(app.exec())

    except Exception as e:
        logging.error(f"Error initializing application: {e}")
        raise(e)

if __name__ == "__main__":
    main()