# License plate recognition

> Perform license plate recognition on video file or webcam

## Installation

To use the script these packages must be installed:

- torch
- torchvision
- opencv
- easyocr
- pymot

Note: It is recommended to use opencv compiled with CUDA support, because algorithms that use opencv's darknet
backend like YOLOv3/v4 run an order of magnitude faster on a GPU.

## Usage

Run the script with the following arguments:

- --lpr_save_img - whether or not to save license plate image to directory saved_number_plates (default is false)
- --video_file_path or -v - use a video for tracking and if the path is not provided use a webcam (default is webcam)

```sh
python3 license_plate_recognition.py --lpr_save_img 0 -v PATH_TO_VIDEO_FILE
```

## License

[MIT](https://choosealicense.com/licenses/mit/)