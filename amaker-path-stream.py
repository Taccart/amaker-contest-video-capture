from typing import List
import cv2
import numpy as np
import logging
import serial
import serial.tools.list_ports
import threading
import datetime
import time
import os
from enum import Enum
os.environ["QT_QPA_PLATFORM"] = "xcb"
class   BotStatus(Enum):
    """Enum for bot states"""
    WAITING=0
    MOVING=1
    SEARCHING=2
    FETCHING=3
    CATCHING=4
    DROPING=5
    STOPPED=6
    TO_SAFETY=10
    MISSON_COMPLETED=20
    
class BotTracker():
    """Class to track a bot's position and color (for identifaction and video feedback)"""
    def __init__(self, name:str="microbot",id:int=None,
                  color_a=(0, 0, 250),
                  color_b=(0, 0, 255),
                  trail_color=(0,0,250),
                  trail_length:int=20):
        self.name = name
        self.id = id    
        self.color_a = color_a
        self.color_b = color_b
        self.trail_color = trail_color
        self.trail_length = trail_length
        self.trail = []
        self.status:BotStatus = BotStatus.WAITING
        self.total_distance=0
    

    def add_position(self, position):
        """Add a new position to the bot's trail"""
        self.trail.append(position)
        if len(self.trail) > self.trail_length:
            self.trail.pop(0) 
        self.total_distance += self.calculate_distance(position)

    def get_last_position(self)->tuple:
        """Get the last known position of the bot"""
        if self.trail:
            return self.trail[-1]
        else:
            return None
    def set_bot_status(self, state:BotStatus):
        """Set the bot's state"""
        self.status = state
        logging.info(f"Bot {self.name}:{self.id} state changed to {self.status.name}")

    def get_bot_status(self)->BotStatus:
        """Get the current state of the bot"""
        return self.status
    
    def calculate_distance(self, position)->float:
        """Calculate the distance from the last position to the current position"""
        if len(self.trail) < 2:
            return 0
        last_position = self.trail[-2]
        distance = np.linalg.norm(np.array(position) - np.array(last_position))
        return distance
    def get_total_distance(self)->float:
        """Get the total distance traveled by the bot"""
        return self.total_distance
    def get_trail(self)->List:
        """Get the bot's trail"""
        return self.trail   
    
    def get_bot_info(self)->str:
        """Get bot information"""
        return f"{self.name}.{self.id} : {self.get_bot_status()}"

    def __repr__(self):
        return f"BotTracker(name={self.name}, color_a={self.color_a}, color_b={self.color_b}, trail_color={self.trail_color}, trail_length={self.trail_length})"
    
class VideoTracker():
    def __init__(self, camera_index:int=0,
                  bot_trackers:List[BotTracker]=None, serial_port=None,
                  
                 baud_rate=None
                   ):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        if baud_rate    is not None:
            logging.info("Serial communication activated.")
            self._init_serial(serial_port, baud_rate)
        else:
            logging.info("Serial communication not activated.")
        
        if not isinstance(bot_trackers, list):
            raise ValueError("No bot trackers provided.")
        if len(bot_trackers) < 1: 
            raise ValueError("No bot trackers provided.")
        self.bot_trackers = bot_trackers
        self.camera_index = self.camera_choice() if camera_index <= 0 else camera_index

        
            
        self.video_capture = cv2.VideoCapture(self.camera_index)  # Use self.camera_index
        cv2.setLogLevel(2)  # Set OpenCV log severity to no logs
        if not self.video_capture.isOpened():
            raise ValueError(f"Camera {camera_index} not found or cannot be opened.")
        
        
        
    def _init_serial(self, serial_port:str=None, baud_rate:int=115200):
        """Initialize serial connection"""
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        self.serial_connection = None
        self.serial_running = False
        self.serial_data = []
        self.serial_thread = None
        try:
            if not self.serial_connection:
                if not self.connect_serial(self.serial_port, self.baud_rate):
                    logging.warning("Serial connection not established. Continuing without serial.")
            
            if self.serial_connection:
                self.serial_thread = threading.Thread(target=self.read_serial)
                self.serial_thread.daemon = True
                self.serial_thread.start()
                logging.info("Serial thread started")
            else:
                logging.info("Serial thread NOT started")                

        except serial.SerialException as e:
            logging.error(f"Failed to connect to serial port: {e}")
            exit()
    ## Get screen resolution - not used
    #screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height
    def camera_choice(self)->int :
        # list all available cameras
        index = 0
        available_cameras = []
        logging.info(f"Searching for cameras...")
        while True:
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    logging.debug(f"Camera {index} : not ok.")
                else:
                    logging.info(f"Camera {index} : ok.")
                    available_cameras.append(index)
                    cap.release()
            except Exception as e:
                logging.warning(f"Camera {index}: error {e}")
            if index > 10:
                break
            index += 1
        if not available_cameras:
            logging.error("No cameras found!")
            exit()

        # choose camera to be used 
        print("\nAvailable cameras:")
        for cam in available_cameras:
            logging.info(f"Camera {cam}")

        selected_camera = int(input("\nSelect the camera index to use: "))
        if selected_camera not in available_cameras:
            logging.error("Invalid camera index selected!")
            exit()
        return selected_camera

    def list_serial_ports(self):
        """List all available serial ports"""
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        print("\nAvailable serial ports:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device} - {port.description}")
            available_ports.append(port.device)
            
        return available_ports
    
    def connect_serial(self, port=None, baud_rate=None):
        """Connect to a serial port"""
        if not port:
            available_ports = self.list_serial_ports()
            if not available_ports:
                logging.warning("No serial ports available")
                return False
                
                logging.info("Available serial ports listed")
            try:
                selection = int(input("\nSelect serial port number (or -1 to skip): "))
                if selection == -1:
                    return False
                port = available_ports[selection]
            except (ValueError, IndexError):
                logging.error("Invalid selection")
                return False
        
        baud_rate = baud_rate or self.baud_rate
        
        try:
            self.serial_connection = serial.Serial(port, baud_rate, timeout=1)
            logging.info(f"Connected to {port} at {baud_rate} baud")
            self.serial_port = port
            self.baud_rate = baud_rate
            return True
        except serial.SerialException as e:
            logging.error(f"Failed to connect to serial port: {e}")
            return False
    
    def read_serial(self):
        """Read data from serial port in a separate thread"""
        self.serial_running = True
        while self.serial_running and self.serial_connection and self.serial_connection.is_open:
            try:
                if self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.readline().decode('utf-8').strip()
                    if data:
                        self.serial_data.append(data)
                        logging.debug(f"Data received: {data}")
                        # Process data here if needed
                time.sleep(0.01)  # Small delay to prevent CPU hogging
            except Exception as e:
                logging.error(f"Serial read error: {e}")
                break

# Mouse callback function
    def button_start(self):
        logging.info("START !")
    def button_stop(self):
        logging.info("STOP !")
    def button_safety(self):
        logging.info("SAFETY !")
        
    def build_display_frame(self, input_frame, is_fullscreen, screen_width, screen_height,window_width, window_height, buttons=None):
        display_frame= input_frame.copy()  # Make a copy for display
        if is_fullscreen:
            display_frame = cv2.resize(input_frame, (screen_width, screen_height))
        else:
            # Use current window dimensions or default to original frame size
            display_frame = cv2.resize(input_frame, (window_width, window_height))
                
        if hasattr(self, 'serial_data') and self.serial_data and len(self.serial_data) > 0:
            serial_data=self.serial_data.pop(0)
            cv2.putText(input_frame, f"Serial: {serial_data}", 
                        (10, input_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            logging.info(serial_data)
        cv2.putText(display_frame, "Unleash The Bricks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2, cv2.LINE_AA)
        for button in buttons:  # Changed from [buttons] to buttons
            cv2.rectangle(display_frame, 
                        (button['x'], button['y']), 
                        (button['x'] + button['w'], button['y'] + button['h']), 
                        (250, 250, 250), 
                        -1)  # Filled rectangle
            cv2.rectangle(display_frame, 
                        (button['x'], button['y']), 
                        (button['x'] + button['w'], button['y'] + button['h']), 
                        (100, 100, 100), 
                        2)   # Border
            cv2.putText(display_frame, 
                    button['text'], 
                    (button['x'] + 10, button['y'] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (50,50,50), 
                    2)
        return display_frame
    def start_tracking(self,recording:bool=False,    window_name = "aMaker microbot tracker"):
        
        ret, input_frame = self.video_capture.read()
        if not ret:
            logging.error("Failed to get initial frame")
            return
        original_height, original_width = input_frame.shape[:2]
        # Create a resizable window with GUI controls
#        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)  # Allow resizing
        #cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)  # Keep aspect ratio
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_OPENGL, 0)  # Disable OpenGL support which can include status bar
        cv2.resizeWindow(window_name, original_width, original_height)  # Start with original frame size
        # Define button dimensions
        button_start = {'x': 5, 'y': 50, 'w': 120, 'h': 40, 'text': 'START', 'clicked': False}
        button_stop = {'x': 5, 'y': 90, 'w': 120, 'h': 40, 'text': 'STOP', 'clicked': False}
        button_safety = {'x': 5, 'y': 130, 'w': 120, 'h': 40, 'text': 'SAFETY', 'clicked': False}
    
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if (button_safety['x'] <= x <= button_safety['x'] + button_safety['w'] and 
                    button_safety['y'] <= y <= button_safety['y'] + button_safety['h']):
                    button_safety['clicked'] = True
                    logging.info("Button start clicked!")
                    # Execute your custom code here
                    self.button_safety()  # Custom method to handle button click
                elif (button_start['x'] <= x <= button_start['x'] + button_start['w'] and 
                    button_start['y'] <= y <= button_start['y'] + button_start['h']):
                    button_start['clicked'] = True
                    logging.info("Button start clicked!")
                    # Execute your custom code here
                    self.button_start()  # Custom method to handle button click
                elif (button_stop['x'] <= x <= button_stop['x'] + button_stop['w'] and
                      button_stop['y'] <= y <= button_stop['y'] + button_stop['h']):
                    button_stop['clicked'] = True
                    logging.info("Button stop clicked!")
                    # Execute your custom code here
                    self.button_stop()
    
        # Set the mouse callback
        cv2.setMouseCallback(window_name, mouse_callback)
        
        
        is_fullscreen = True
        window_width, window_height = original_width, original_height  # Initialize with frame size

        cv2.setNumThreads(7)  # Set the number of threads for OpenCV
        #cv2.setUseOptimized(True)  # Enable OpenCV optimizations
        try:
        # Try to get screen info - this is optional and only works if screeninfo is installed
            from screeninfo import get_monitors
            monitor = get_monitors()[0]
            screen_width, screen_height = monitor.width, monitor.height
            logging.info(f"Detected screen size: {screen_width}x{screen_height}")
        except:
            # Fallback to reasonable defaults if we can't get screen info
            screen_width, screen_height = 1280, 960
            logging.warning(f"Couldn't detect screen size. Using defaults: {screen_width}x{screen_height}")
    

        video_writer = None
        if recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name=f'aMaker_microbot_tracker_{current_time}.avi'
            video_writer = cv2.VideoWriter(video_name, fourcc, 20.0, (1280, 960))
            logging.info(f"Video recording started : {video_name}")
        else:
            logging.info("No video recording.")
        
        logging.info(f"Bot trackers: {self.bot_trackers}")
        logging.info(f"Press 'q' or ESC to quit.")
        logging.info(f"Press 'f' to toggle full screen.")
    
        
        while True:
            ret, input_frame = self.video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break


            
                   
            # Convert to HSV for color detection
            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGRA2RGB)
            display_frame= self.build_display_frame(input_frame, is_fullscreen, screen_width, screen_height,window_width, window_height, [button_start, button_stop, button_safety])

            for bot_tracker in self.bot_trackers:
                # Get the bot's color range
                bot_mask = cv2.inRange(rgb, bot_tracker.color_a, bot_tracker.color_b)    
                bot_contours, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
        
                # Extract bot positions
                if bot_contours:
                    x, y, w, h = cv2.boundingRect(bot_contours[0])
                    bot_position = (x + w//2, y + h//2)
                    bot_tracker.add_position(bot_position   )
                    
                    if len(bot_tracker.get_trail()) > 1:
                        cv2.polylines(input_frame, [np.array(bot_tracker.get_trail())], False, bot_tracker.trail_color, 2)
                        cv2.circle(input_frame, bot_position, 5, bot_tracker.trail_color, -1)
                        cv2.rectangle(input_frame, (x, y), (x + w, y + h), bot_tracker.trail_color, 2)
                        #cv2.putText(frame, bot_tracker.name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bot_tracker.trail_color, 2)
                    
                    
        

            # Add text to the frame
            
            if video_writer:
                # Save the frame to the video file
                video_writer.write(input_frame)
            
            # Show the video feed

            
            cv2.putText(display_frame, bot_tracker.get_bot_info(), (50,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bot_tracker.trail_color, 2)
              # Display serial data on frame if available
        
            cv2.imshow(window_name, )
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 4 or key == 27:  # 'q' or Ctrl+D or ESC to quit
                break
            #elif key == ord('f'):  # 'f' to toggle fullscreen
            #    is_fullscreen = not is_fullscreen
            #    if is_fullscreen:
            #        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            #        logging.info("Fullscreen mode enabled")
            #        logging.info(f"old screen size: {screen_width}x{screen_height}")
            #        screen = cv2.getWindowImageRect(window_name)
            #        
            #        x, y, old_width, old_height = screen
            #        logging.info(f"new screen size: {x}x{y} {old_width}x{old_height}")
            #    else:
            #        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            #        screen_width, screen_height = None, None
            #
            #        logging.info("Fullscreen mode disabled")
                
        if hasattr(self, 'serial_connection') and self.serial_connection and self.serial_connection.is_open:
            self.serial_running = False
            if self.serial_thread:
                self.serial_thread.join(timeout=1.0)
            self.serial_connection.close()
            logging.info("Serial connection closed")
        self.video_capture.release()
        if video_writer:
            # Release the video writer
            video_writer.release()
            print("Video saved successfully.")
            
        def __del__(self):
            """Destructor cleans up resources"""        
            try:
                if hasattr(self, 'serial_connection') and self.serial_connection and self.serial_connection.is_open:
                    if self.serial_thread:
                        self.serial_thread.join(timeout=1.0)
                    self.serial_running = False
                    self.serial_connection.close()
                    
                    logging.info("Serial connection closed")       
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")                    
            try:
                if hasattr(self, 'video_capture') and self.video_capture.isOpened():
                    self.video_capture.release()
                    logging.info("Video capture released")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")                    
            try:
                cv2.destroyAllWindows()
                logging.info("All windows destroyed")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")                    
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
 

if __name__ == "__main__":
    
    bot1 = BotTracker(name="bot1", color_a=(160, 100, 100), color_b=(255, 110, 100), trail_length=10)
    vt = VideoTracker( camera_index=4,bot_trackers=[bot1], serial_port="/dev/ttyACM0", baud_rate=57600)
    vt.start_tracking(recording=False)