import argparse
import logging
import logging.handlers
import os
import sys
import time
from typing import Dict, Any

import cv2
import yaml
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QPixmap, QImage, QAction, QColor
# PyQt imports - grouped together and explicitly formatted
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QDockWidget, QMenuBar, QSizePolicy)

from amaker.communication.serial_communication_manager import SerialCommunicationManagerImpl
from amaker.unleash_the_bricks import DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT, VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT, \
    UI_RGBCOLOR_BRIGHTGREEN
from amaker.unleash_the_bricks.bot import UnleashTheBrickBot
from amaker.unleash_the_bricks.controller import AmakerBotTracker

# Set environment variable for Qt
os.environ["QT_QPA_PLATFORM"] = "xcb"

#
VIDEO_REFRESH_FPS = 20

DEFAULT_WINDOW_MAIN_X = 100
DEFAULT_WINDOW_MAIN_Y = 100
DEFAULT_WINDOW_MAIN_WIDHT = 1280
DEFAULT_WINDOW_MAIN_HEIGHT = 1024

DEFAULT_WINDOW_LOG_X = 10
DEFAULT_WINDOW_LOG_Y = 50
DEFAULT_WINDOW_LOG_WIDHT = 500
DEFAULT_WINDOW_LOG_HEIGHT = 400

DEFAULT_WINDOW_BOTS_X = 60
DEFAULT_WINDOW_BOTS_Y = 10
DEFAULT_WINDOW_BOTS_WIDHT = 1024
DEFAULT_WINDOW_BOTS_HEIGHT = 100
DEFAULT_WINDOW_BOTS_MAX_SHOWN = 6

REFRESH_LOG_MS = 250
REFRESH_TABLE_MS = 250

VIDEO_DEFAULT_BOT_COLOR = UI_RGBCOLOR_BRIGHTGREEN
VIDEO_MIN_X = 640
VIDEO_MIN_Y = 480
UI_SYSTEM_LOG_MAX_LINES = 100
UI_SYSTEM_LOG_FORMAT = '%(asctime)s %(levelname).3s %(message)s'


class LogDockWidget(QDockWidget):
    def __init__(self):
        super().__init__("Logs")
        self._LOG = logging.getLogger(__name__)
        self.setGeometry(DEFAULT_WINDOW_LOG_X, DEFAULT_WINDOW_LOG_Y, DEFAULT_WINDOW_LOG_WIDHT,
                         DEFAULT_WINDOW_LOG_HEIGHT)

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
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable)

    def append_log(self, message):
        self.logs_text.append(message)

    def update_text(self, logs: list[str] | None):
        if logs is None:
            self.logs_text.setText("")
        elif len(logs) > 0:
            self.logs_text.setText("\n".join(logs))
        else:
            self.logs_text.setText("")


class BotsDockWidget(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bots")
        self.setGeometry(DEFAULT_WINDOW_BOTS_X, DEFAULT_WINDOW_BOTS_Y, DEFAULT_WINDOW_BOTS_WIDHT,
                         DEFAULT_WINDOW_BOTS_HEIGHT)
        # Initialize with 0 rows, will resize dynamically
        self.table_widget = QTableWidget(0, 5)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setHorizontalHeaderLabels(["Team", "bot state", "distance", "self est. count", ""])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setWidget(self.table_widget)

        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable)

    def update_table(self, bots: list[UnleashTheBrickBot]):
        current_row_count = self.table_widget.rowCount()
        bots_count = len(bots)

        if current_row_count != bots_count:
            self.table_widget.setRowCount(bots_count)

        for row, bot in enumerate(bots):
            self.table_widget.setItem(row, 0, QTableWidgetItem(bot.name))
            self.table_widget.setItem(row, 1, QTableWidgetItem(str(bot.status)))
            self.table_widget.setItem(row, 2, QTableWidgetItem(f"{bot.total_distance / 100:6.2f}"))
            self.table_widget.setItem(row, 3, QTableWidgetItem(str(bot.collected_count)))
            colorCell = QTableWidgetItem(" ")
            r, g, b = bot.color
            colorCell.setBackground(QColor(r, g, b))

            self.table_widget.setItem(row, 4, colorCell)
        # Resize window to fit table contents
        self.resize_window_to_fit_table()

    def resize_window_to_fit_table(self):
        header_height = self.table_widget.horizontalHeader().height()
        row_height = 30
        total_row_height = 0
        for i in range(self.table_widget.rowCount()):
            total_row_height += self.table_widget.rowHeight(i) or row_height
        padding = 40


        new_height = header_height + total_row_height + padding
        current_width = self.width()

        min_height = 100
        new_height = max(new_height, min_height)

        self.resize(current_width, new_height)


class UILogHandler(logging.Handler):
    def __init__(self, log_window, max_lines=UI_SYSTEM_LOG_MAX_LINES):
        super().__init__()
        self.log_window = log_window
        self.max_lines = max_lines
        self.log_buffer = []
        self.setFormatter(logging.Formatter(UI_SYSTEM_LOG_FORMAT, datefmt='%H:%M:%S'))

    def emit(self, record):
        log_entry = self.format(record)
        self.log_buffer.append(log_entry)

        if len(self.log_buffer) > self.max_lines:
            self.log_buffer = self.log_buffer[-self.max_lines:]

        self.log_window.update_text(self.log_buffer)


class AmakerControllerUI(QMainWindow):
    def __init__(self, controller: AmakerBotTracker, max_tracked_bots_count=4):
        super().__init__()
        self._LOG = logging.getLogger(__name__)
        self.setWindowTitle("Unleash The Bricks")
        self.setGeometry(DEFAULT_WINDOW_MAIN_X, DEFAULT_WINDOW_MAIN_Y, DEFAULT_WINDOW_MAIN_WIDHT,
                         DEFAULT_WINDOW_MAIN_HEIGHT)
        self.max_tracked_bots_count = max_tracked_bots_count
        self.controller: AmakerBotTracker = controller



        # Create menu bar : Order and View
        self.menu = self.menuBar()
        self.setup_menu_bar(self.menu)
        self.setup_docking_windows()

        main_layout = QVBoxLayout()

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


        # Video stream
        self.video_place_holder = QLabel()
        self.video_place_holder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_place_holder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_place_holder.setMinimumSize(VIDEO_MIN_X, VIDEO_MIN_Y)  # Set a reasonable minimum size

        main_layout.addWidget(self.video_place_holder)

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

        self.communication_log_timer = QTimer()
        self.communication_log_timer.timeout.connect(self.update_communication_log)
        self.communication_log_timer.start(REFRESH_LOG_MS)  # Update table every 500
        # Restore window state if available
        self.settings = QSettings("AmakerBot", "UnleashTheBricks")
        if self.settings.contains("mainWindowGeometry"):
            self.restoreGeometry(self.settings.value("mainWindowGeometry"))
        if self.settings.contains("mainWindowState"):
            self.restoreState(self.settings.value("mainWindowState"))

    def setup_docking_windows(self):

        # Create and add dock widgets
        self.bots_info_window = BotsDockWidget()
        self.bots_info_window.setWindowTitle("Bots")
        self.bots_info_window.setObjectName("bots_info_window")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bots_info_window)
        self.bots_info_window.visibilityChanged.connect(self.toggle_bot_info_action)

        # Create communication logs window (if it doesn't exist)
        self.comm_logs_window = LogDockWidget()
        self.comm_logs_window.setWindowTitle("Communication logs")
        self.comm_logs_window.setObjectName("communication_logs_window")
        self.tabifyDockWidget(self.bots_info_window, self.comm_logs_window)
        self.comm_logs_window.visibilityChanged.connect(self.toggle_communication_logs)

        self.system_logs_window = LogDockWidget()
        self.system_logs_window.setWindowTitle("System logs")
        self.system_logs_window.setObjectName("system_logs_window")
        self.tabifyDockWidget(self.bots_info_window, self.system_logs_window)
        self.system_logs_window.visibilityChanged.connect(self.toggle_system_logs_action)
        self.log_handler = UILogHandler(self.system_logs_window, max_lines=100)
        self.log_handler.setLevel(logging.INFO)  # Set appropriate level
        logging.getLogger().addHandler(self.log_handler)


    def setup_menu_bar(self, menu_bar: QMenuBar):

        self.order_menu = menu_bar.addMenu("Order")
        self.view_menu = menu_bar.addMenu("View")

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # If there's a pixmap, update its scaling
        if hasattr(self.video_place_holder, 'pixmap') and not self.video_place_holder.pixmap().isNull():
            self.update_video_place_holder_pixmap(self.video_place_holder.pixmap())

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

    def toggle_communication_logs(self, visible: bool):
        self.comm_logs_action.setChecked(visible)

    def update_communication_log (self):
        if self.controller:
            while True:
                data = self.controller.communication_manager.get_next_data()
                if data is None:
                    break
                self.controller._add_communication_log(data)
            self.comm_logs_window.update_text(self.controller.communication_logs)

    def update_log(self):
        if self.controller:
            self.comm_logs_window.update_text(self.controller.logs)

    def update_video_place_holder_pixmap(self, pixmap):
        """Scale the video frame to fill the available space while preserving aspect ratio"""
        if pixmap.isNull():
            return

        # Scale the pixmap to fill the label while maintaining aspect ratio
        label_size = self.video_place_holder.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.video_place_holder.setPixmap(scaled_pixmap)

    def update_frame(self):
        if self.controller:

            ret, frame = self.controller.video_capture.read()
            if ret:
                # Process the frame using the controller
                detected_tags = self.controller.bot_tracker.detect(frame)
                self.controller.overlay_tags(frame, detected_tags)
                # deactivate bots infos to fasten frame
                self.controller.amaker_ui.show_bot_infos(frame, self.controller.tracked_bots)

                self.controller.amaker_ui.ui_add_countdown(frame, self.controller.deadline)

                self.controller.process_pending_feed_messages()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                q_img = QImage(frame.data, width, height, step, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.update_video_place_holder_pixmap(pixmap)
                if self.controller.video_writer and self.controller.video_writer.isOpened():
                    # Save the frame to the video file
                    video_out = cv2.resize(frame, (VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT))
                    self.controller.video_writer.write(video_out)
                try:
                    self.controller.process_pending_feed_messages()
                except Exception as e:
                    self._LOG.error(f"Error processing feed messages: {e}")


    def update_table(self):
        if self.controller and hasattr(self.controller, 'tracked_bots'):
            bots = list(self.controller.tracked_bots.values())
            self.bots_info_window.update_table(bots)

    def start_button_clicked(self):
        if self.controller:
            self.controller.on_UI_BUTTON_start()
        else:
            self._LOG.error("Controller is not initialized. Cannot handle start.")

    def stop_button_clicked(self):
        if self.controller:
            self.controller.on_UI_BUTTON_stop()
        else:
            self._LOG.error("Controller is not initialized. Cannot handle stop.")

    def safety_button_clicked(self):
        if self.controller:
            self.controller.on_UI_BUTTON_safety()
        else:
            self._LOG.error("Controller is not initialized. Cannot handle safety.")

    def closeEvent(self, event):
        if hasattr(self, 'log_handler'):
            logging.getLogger().removeHandler(self.log_handler)

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
    parser.add_argument('--config', metavar='path', required=True, help='Path to YAML configuration file', type=str)
    return parser.parse_args()

def load_and_validate_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file and apply defaults/validation"""
    try:
        # Load config from file
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Define default values
        defaults = {"window": {"width": DEFAULT_SCREEN_WIDTH, "height": DEFAULT_SCREEN_HEIGHT},
                    "camera": {"calibration_file": "TALogitechHDWebcamB910_calibration.npz", "number": -1},
                    "communication": {"serial_port": "/dev/ttyACM0", "serial_speed": 57600},
                    "recording": {"feed_enabled": False, "path": "./"}, "logs": {"max_count": 10, "show_on_screen": False}}
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
            if "path" not in config["recording"]:
                config["recording"]["path"] = defaults["recording"]["path"]

        config["recording"]["feed_enabled"] = config["recording"].get("feed_enabled", defaults["recording"]["feed_enabled"])
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')

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

            if tag_type == "bot" and tag_info.get("name") in config["session_options"]["participants"]:

                tracked_bots[tag_id] = UnleashTheBrickBot(name=tag_info.get("name", f"Bot {tag_id}"), bot_id=tag_id,
                                                          rgb_color=tuple(tag_info.get("color",
                                                                                       VIDEO_DEFAULT_BOT_COLOR)) if tag_info.get(
                                                              "color") else VIDEO_DEFAULT_BOT_COLOR, )
            else:
                reference_tags[tag_id] = {"name": tag_info.get("name", ""), "type": tag_type}

        # Select camera
        camera_number = config["camera"]["number"]
        if camera_number < 0:
            camera_number = AmakerBotTracker.user_input_camera_choice()
        # Create communication manager
        communication_manager = SerialCommunicationManagerImpl(baud_rate=config["communication"]["serial_speed"],
                                                               serial_port=config["communication"]["serial_port"])
        # Initialize the bot tracker
        controller = AmakerBotTracker(calibration_file=config["camera"]["calibration_file"], camera_index=camera_number,
                                      communication_manager=communication_manager, tracked_bots=tracked_bots,
                                      window_size=(config["window"]["width"], config["window"]["height"]),
                                      know_tags=reference_tags, max_logs=config["logs"]["max_count"],
                                      countdown_seconds=config["session_options"]["countdown_second"], info_feed_interval_second= config["session_options"]["feed_interval_second"]

                                      )

        window = AmakerControllerUI(controller=controller)

        window.controller.amaker_ui.is_overlay_logs = config["logs"]["show_on_screen"]

        # Start video timer
        window.video_timer.start(VIDEO_REFRESH_FPS)
        controller.communication_manager.start_reading()
        time.sleep(0.2)

        # Show the window
        window.show()

        # Configure video recording if feed_enabled
        if config["recording"]["feed_enabled"]:
            if config["recording"]["path"]:
                if os.path.isdir(config["recording"]["path"]):
                    window.controller.setup_video_recording(True, config["recording"]["path"])
                else:
                    logging.error(f"NO RECORDING  - Invalid recording path: {config['recording']['path']}")

        # Run the application
        sys.exit(app.exec())

    except Exception as e:
        logging.error(f"Error initializing application: {e}")
        raise (e)

if __name__ == "__main__":
    main()
