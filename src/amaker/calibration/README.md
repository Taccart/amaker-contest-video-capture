# Camera calibration
The application uses OpenCV for video capture and processing.
OpenCV requires a camera calibration to be fully operational. 
Read more information at [docs.opencv.org: tutorial py calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

To run the calibration, print the chessboard pattern [chessboard 7x9 with 20x20 mm squares.pdf](./chessboard 7x9 with 20x20 mm squares.pdf)
and take multiple pictures of it with your camera, under many angles and distances.

***********************************************************************************
IT'S MANDATORY TO HAVE THE CHESSBOARD PRINTED SHEET FLAT AND WITHOUT ANY FOLDS.
***********************************************************************************


The calibration.py script will create a .npz file with the camera matrix and distortion coefficients.

This file will be used to undistort images taken with THIS PARTICULAR CAMERA.



### Camera Calibration Command-Line Arguments

The calibration script supports the following command-line arguments:

| Argument               | Description                                      | Type    | Default Value         |
|------------------------|--------------------------------------------------|---------|-----------------------|
| `--chessboard_rows`    | Number of inner corners per row in the chessboard | Integer | 9                     |
| `--chessboard_cols`    | Number of inner corners per column in the chessboard | Integer | 6                     |
| `--square_size`        | Size of a square in your chessboard (in any unit, e.g., mm) | Float   | 1.0                   |
| `--image_dir`          | Directory containing chessboard images           | String  | `./calibration_images`|
| `--output_file`        | Path to save the calibration file                | String  | `camera_calibration.npz` |

### Example Usage
Show help message:
```bash
python calibration.py -h
```

Run calibration with default parameters corresponding to chessboard 7x9 with 20x20 mm squares (as in given pdf):
```bash
python calibration.py
```

`
Run calibration with specific parameters:
```bash
python calibration.py --chessboard_rows 7 --chessboard_cols 5 --square_size 25.0 --image_dir ./my_image_dir --output_file my.npz
```