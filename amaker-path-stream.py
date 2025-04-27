import cv2
import numpy as np
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
cv2.setLogLevel(0)  # Set OpenCV log severity to no logs

# Define colors for tracking
BOT1_COLOR = (0, 0, 255)  # RED

## Get screen resolution - not used
#screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height
def camera_choice()->int :
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
        print("No cameras found!")
        exit()

    # choose camera to be used 
    print("\nAvailable cameras:")
    for cam in available_cameras:
        print(f"Camera {cam}")

    selected_camera = int(input("\nSelect the camera index to use: "))
    if selected_camera not in available_cameras:
        print("Invalid camera index selected!")
        exit()
    return selected_camera


# Open the selected camera
cap = cv2.VideoCapture(4)#(camera_choice())

# Lists to store positions
bot1_positions = []
# Create a resizable window with GUI controls
cv2.namedWindow("Bot Tracking", cv2.WINDOW_GUI_EXPANDED)  # Allow resizing
cv2.setWindowProperty("Bot Tracking", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)  # Keep aspect ratio
cv2.setNumThreads(8)  # Set the number of threads for OpenCV
cv2.setUseOptimized(True)  # Enable OpenCV optimizations
fourcc = cv2.VideoWriter_fourcc(*'XVID')
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_writer = cv2.VideoWriter(f'bot_tracker_{current_time}.avi', fourcc, 20.0, (640, 480))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    # Convert to HSV for color detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Thresholding to detect colors (you'll need to adjust these ranges)
    bot1_mask = cv2.inRange(rgb, (250, 250, 250),(255,255,255))  # red 
    # Find contours (Bot positions)0
    bot1_contours, _ = cv2.findContours(bot1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # Extract bot positions
    if bot1_contours:
        x, y, w, h = cv2.boundingRect(bot1_contours[0])
        bot1_positions.append((x + w//2, y + h//2))
    if len(bot1_positions) > 10:
        bot1_positions.pop(0)



    # Draw paths
    if len(bot1_positions) > 1:
        cv2.polylines(frame, [np.array(bot1_positions)], False, BOT1_COLOR, 2)
        #cv2.circle (frame, bot1_positions[-1], 5, BOT1_COLOR, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOT1_COLOR, 2)


    video_writer.write(frame)
    # Add text to the frame
    cv2.putText(frame, "Press Q to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BOT1_COLOR, 2, cv2.LINE_AA)
    # Show the video feed
    cv2.imshow("Bot Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()

cv2.destroyAllWindows()
