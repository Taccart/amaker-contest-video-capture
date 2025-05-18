# amaker-contest-video-capture
##  BEFORE YOU START

This code relies on opencv and other libraries. OpenCV is a powerful library for computer vision tasks, and it is essential for the functionality of this code. 

YOU HAVE TO CREATE THE CALIBRATION FOR YOUR CAMERA, YOURSELF.

Check the [calibration.py](amaker/calibration/calibration.py) file for instructions on how to calibrate your camera.

```bash

## Unleash The Bricks

aMaker microbot tournament tracking and visualization software.

### Installation

```bash
pip install amaker-unleash-the-bricks
```

### Usage

```bash
unleash-the-bricks --camera_number 0
```

### Command-Line Arguments

The application supports the following command-line arguments:

| Argument               | Description                                      | Type    | Default Value         |
|------------------------|--------------------------------------------------|---------|-----------------------|
| `--window_width`       | Window width                                     | Integer | 1920 (default screen width) |
| `--window_height`      | Window height                                    | Integer | 1080 (default screen height) |
| `--camera_calibration_file` | Path to the camera calibration file          | String  | `camera_calibration.npz` |
| `--camera_number`      | Camera number. Use `-1` to select from a list.   | Integer | -1                    |
| `--serial_port`        | Serial port (e.g., `/dev/ttyACM0`)               | String  | `/dev/ttyACM0`        |
| `--serial_speed`       | Serial port baud rate (e.g., 57600)              | Integer | 57600                 |
| `--record_video`       | Enable video recording                           | Boolean | False                 |
| `--record_video_path`  | Path to save recorded videos                     | String  | `./`                  |

### Example Usage

```bash
unleash-the-bricks --camera_number 0 --window_width 1280 --window_height 720 --record_video True --record_video_path ./videos/