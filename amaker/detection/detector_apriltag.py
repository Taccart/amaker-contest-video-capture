from typing import List

import cv2
import numpy as np
from pyapriltags import Detector, Detection

from amaker.detection.detector_abstract import DetectorAbstract

DEFAULT_TAG_SIZE_CM = 10  # cm


class AprilTagDetectorImpl(DetectorAbstract):
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

    def detect(self, *args, **kwargs) -> List[Detection]:
        """
         Detect AprilTags in a frame.

         Args:
             *args: First arg should be the frame
             **kwargs: Optional parameters including:
                 - tag_size_cm: Tag size in cm (default: 10.0)
                 - skip_undistort: Skip undistortion for speed (default: False)
                 - detect_pose: Whether to estimate pose (default: True)

         Returns:
             List of Detection objects
        """

        if not args or len(args) < 1:
            raise ValueError("Frame is required as the first argument.")

        frame = args[0]
        if frame is None:
            raise ValueError("Frame is required as the first argument.")
        tag_size_m = int(kwargs.get('tag_size_cm', DEFAULT_TAG_SIZE_CM)) / 100
        skip_undistort = kwargs.get('skip_undistort', False)
        detect_pose = kwargs.get('detect_pose', True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply undistortion if calibration is available and not skipped

        # Run detection

        detected = self.apriltag_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=tag_size_m
        )

        return detected
