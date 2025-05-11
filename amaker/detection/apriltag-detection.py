import logging
import time
from typing import List

import cv2
import numpy as np
from pyapriltags import Detector, Detection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
COLOR_IGNORED = (255, 255, 255)
COLOR_WALL = (255, 255, 100)
COLOR_GROUND = (200, 255, 100)
COLOR_BOT = (100, 100, 255)
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
DETECTOR_THREADS = 4
CV_THREADS = 4
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


def identify_apriltags(frame, mtx, dist, fx, fy, cx, cy, tag_size=0.10) -> List[Detection]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_undistorded = cv2.undistort(gray, mtx, dist, None, newCameraMatrix=mtx)
    detected = at_detector.detect(gray_undistorded, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy],
                                  tag_size=tag_size)
    return detected


def compute_angle(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(np.array([x, y, z]))


at_detector = Detector(families="tag36h11", nthreads=DETECTOR_THREADS, quad_sigma=0.0, refine_edges=1,
                       decode_sharpening=0.25, debug=0)
calibration_data = np.load('camera_calibration.npz')
mtx = calibration_data['camera_matrix']
dist = calibration_data['dist_coeffs']
fx = mtx[0, 0]  # Focal length in x direction
fy = mtx[1, 1]  # Focal length in y direction
cx = mtx[0, 2]  # Principal point x-coordinate (optical center)
cy = mtx[1, 2]  # Principal point y-coordinate (optical center)

# Acquisition de l'image, correction et détection des tags présents sur l'image
cv2.setLogLevel(0)  # Set OpenCize.width>0 && size.hV log severity to no logs
cv2.setNumThreads(CV_THREADS)  # Set the number of threads for OpenCV
cv2.setUseOptimized(True)  # Enable OpenCV optimizations
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
logging.info(f"Camera resolution: {actual_width}x{actual_height}")
t0 = time.time()
matrice = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
try:
    while True:
        ret, frame = cap.read()
        # cv2.imshow('Input video', frame)
        color_img = frame.copy()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img_undistorded = cv2.undistort(gray, mtx, dist, None, newCameraMatrix=mtx)
        # tags=at_detector = Detector(families="tag36h11",nthreads=6,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0,camera_params=[fx,fy,cx,cy], tag_size=0.03)
        ##at_detector.detect(img_undistorded, estimate_tag_pose=True,    camera_params=[fx,fy,cx,cy], tag_size=0.03)

        tags = identify_apriltags(frame, mtx, dist, fx, fy, cx, cy)
        if not ret:
            logging.error("Failed to capture frame")
            break
        positions = []
        positionMoyenne = np.array([0, 0, 0], dtype='float64')
        angles = []
        angleMoyen = np.array([0, 0, 0], dtype='float64')

        for tag in tags:
            if tag.tag_id in reference_tags["wall"]:
                for idx in range(len(tag.corners)):
                    cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)), COLOR_GROUND, 5)
                    cv2.putText(color_img, reference_tags["wall"][tag.tag_id]["name"],
                                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_GROUND, thickness=1)

            elif tag.tag_id in reference_tags["ground"]:
                for idx in range(len(tag.corners)):
                    cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)), COLOR_GROUND, 5)
                    cv2.putText(color_img, reference_tags["ground"][tag.tag_id]["name"],
                                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_GROUND, thickness=1)

            elif tag.tag_id in reference_tags["bot"]:

                for idx in range(len(tag.corners)):
                    # cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)),tuple(tag.corners[idx, :].astype(int)), COLOR_BOT,5)
                    cv2.putText(color_img, reference_tags["bot"][tag.tag_id]["name"],
                                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) - 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_BOT, thickness=1)

                    # Draw the 3D axes
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
                        image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
                        image_points = np.int32(image_points).reshape(-1, 2)

                        # Draw the axes with thicker lines and clear colors
                        origin = tuple(image_points[0])
                        cv2.line(color_img, origin, tuple(image_points[1]), (0, 0, 255), 3)  # X-axis (red)
                        cv2.line(color_img, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y-axis (green) 
                        cv2.line(color_img, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z-axis (blue)

                        # Add labels to the axes for better visibility
                        cv2.putText(color_img, "X", tuple(image_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                    2)
                        cv2.putText(color_img, "Y", tuple(image_points[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                    2)
                        cv2.putText(color_img, "Z", tuple(image_points[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                    2)

                        # Log axis points for debugging
                        # logging.debug(f"Axis points: Origin={image_points[0]}, X={image_points[1]}, Y={image_points[2]}, Z={image_points[3]}")
                    else:
                        logging.error(f"Tag {tag.tag_id} missing pose information")

                    logging.info(f"BOT {tag.tag_id},  Position : {tag.pose_t}")


            else:
                for idx in range(len(tag.corners)):
                    cv2.putText(color_img, str(tag.tag_id),
                                org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_IGNORED, thickness=1)

                    cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             COLOR_IGNORED, 5)
            # logging .info(f"Tag ID: {tag.tag_id}, Anle : {compute_angle(tag.pose_R)}, Position : {tag.pose_t}")
            # angles.append(np.array(compute_angle(tag.pose_R)))
            # pose=np.dot(np.transpose(tag.pose_R),tag.pose_t)
            # try :
            #    positions.append(np.dot(matrice,np.transpose(pose)[0]+np.array(reference_tags_positions[tag.tag_id])))
            # except :
            #    print("tag inconnu detecte : ",tag.tag_id)

            t1 = time.time()
            cv2.putText(color_img, "took " + "{:.3f}".format(t1 - t0) + "s", org=(10, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=COLOR_IGNORED, thickness=2)

            t0 = t1
        cv2.imshow('Detected tags', color_img)
        # for position in positions:
        #    positionMoyenne+=position
        # for angle in angles:
        #    angleMoyen+=angle
        # n=len(positions)
        # if n!=0:
        #    positionMoyenne=positionMoyenne/n
        #    print("position et nb tags : ", positionMoyenne,n)
        #    angleMoyen=angleMoyen/n
        #    print("angle : ", angleMoyen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
except Exception as e:

    logging.error(f"An error occurred: {e.with_traceback}")
    raise (e)

finally:
    cap.release()
    cv2.destroyAllWindows()
