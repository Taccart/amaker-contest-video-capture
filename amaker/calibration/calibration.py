import glob
import logging

import cv2 as cv
import numpy as np


class CameraCalibration:
    """
    Class to calibrate a camera using chessboard images.
    To calibrate the camera, you need to provide a set of chessboard images taken from different angles and distances.
    """

    def __init__(self, chessboard_images_path: str = './*.jpg', chessboard_size=(10, 7),
                 destination_file="camera_calibration.npz", wait_time_ms=250):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.debug("CameraCalibration initialized")
        self.source_images_path = chessboard_images_path
        self.destination_file = destination_file
        self.wait_time_ms = wait_time_ms
        self.chessboard_size = chessboard_size  # Number of inner corners per chessboard row and column

    def calibrate_camera(self, square_size_mm: float = 30.0):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, square_size_mm, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros(((self.chessboard_size[0] - 1) * (self.chessboard_size[1] - 1), 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0] - 1, 0:self.chessboard_size[1] - 1].T.reshape(-1, 2)
        logging.info(f"Searching in {self.source_images_path}")
        # Arrays to store object points and image points from all the images.
        object3D_points = []  # 3d point in real world space
        image2D_points = []  # 2d points in image plane.
        try:
            images = glob.glob(self.source_images_path)
            logging.info(f"Source images: {images}")

            if len(images) == 0:
                logging.error("No images found in the specified path: " + str(self.source_images_path))
                raise FileNotFoundError

            for fname in images:
                logging.info("Processing image: " + str(fname))
                img = cv.imread(fname)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray,
                                                        (self.chessboard_size[0] - 1, self.chessboard_size[1] - 1),
                                                        None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    logging.info("Chessboard found with corners are " + str(corners))
                    object3D_points.append(objp)
                    corners2 = cv.cornerSubPix(gray, corners, (self.chessboard_size[0], self.chessboard_size[1]),
                                               (-1, -1), criteria)
                    image2D_points.append(corners2)
                    # Draw and display the corners
                    cv.imshow('Scanning for chessboard', img)
                    cv.waitKey(self.wait_time_ms)
                    img_corners = img.copy()
                    cv.drawChessboardCorners(img_corners, (self.chessboard_size[0] - 1, self.chessboard_size[1] - 1),
                                             corners2, ret)
                    cv.imshow('Scanning for chessboard', img_corners)
                    cv.waitKey(self.wait_time_ms)  # Wait longer to see the corners clearly
                else:
                    logging.warning(f"Failed scamning for a {self.chessboard_size} chessboard from {fname}")

        except Exception as e:
            logging.error(f"Error during images interpretation for calibration: {e}")
        finally:
            cv.destroyAllWindows()

        # Only calibrate if we found corners
        if len(object3D_points) > 0:
            logging.info(f"Calibrating camera with {len(object3D_points)} objects and {len(image2D_points)} images...")

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                object3D_points, image2D_points, gray.shape[::-1], None, None
            )
            logging.info("Camera matrix (mtx):")
            logging.info(mtx)
            logging.info("Distortion coefficients (dist):")
            logging.info(dist)

            # Save the calibration results
            np.savez(self.destination_file,
                     camera_matrix=mtx,
                     dist_coeffs=dist,
                     rvecs=rvecs,
                     tvecs=tvecs)
            logging.info(f"Calibration saved to {self.destination_file}")
        else:
            logging.warning("Calibration failed. No chessboard corners were detected.")


if __name__ == "__main__":
    calibration = CameraCalibration(chessboard_images_path="/home/taccart/Pictures/Webcam/*.jpg")
    calibration.calibrate_camera(30)
