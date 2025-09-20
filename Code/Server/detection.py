import argparse
import sys
import time
import cv2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# import your Camera class
from picamera2 import Picamera2
from libcamera import Transform
import utils


class Camera:
    def __init__(self, preview_size=(640, 480), hflip=False, vflip=False):
        """Initialize the PiCamera2 for object detection."""
        self.camera = Picamera2()
        self.transform = Transform(hflip=1 if hflip else 0, vflip=1 if vflip else 0)

        # Configure camera for preview
        preview_config = self.camera.create_preview_configuration(
            main={"size": preview_size}, transform=self.transform
        )
        self.camera.configure(preview_config)
        self.camera.start()

    def get_frame(self):
        """Capture a single frame and return it as a NumPy array (OpenCV format)."""
        frame = self.camera.capture_array()
        return frame  # already numpy array in RGB


def run(model: str, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
    """Continuously run inference on images from PiCamera2."""

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Initialize PiCamera2
    cam = Camera(preview_size=(width, height))

    # Visualization params
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize TFLite object detection
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                            score_threshold=0.3,
                                            max_results=3)
    detector = vision.ObjectDetector.create_from_options(options)

    # Capture loop
    while True:
        # Get frame from PiCamera2 (RGB numpy array)
        image = cam.get_frame()
        if image is None:
            sys.exit("ERROR: Unable to read from PiCamera2.")

        counter += 1

        # Convert to OpenCV BGR format for display
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create TensorImage from RGB array
        import mediapipe as mp
        input_tensor = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

        # Run inference
        detection_result = detector.detect(input_tensor)

        # Visualize detection results
        bgr_image = utils.visualize(bgr_image, detection_result)

        # FPS calculation
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show FPS
        fps_text = "FPS = {:.1f}".format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(bgr_image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        # Display output
        cv2.imshow("object_detector", bgr_image)

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Path of the object detection model.",
        required=False,
        default="efficientdet_lite0.tflite"
    )
    parser.add_argument(
        "--frameWidth", help="Width of frame.", type=int, default=640
    )
    parser.add_argument(
        "--frameHeight", help="Height of frame.", type=int, default=480
    )
    parser.add_argument(
        "--numThreads", help="Number of CPU threads.", type=int, default=4
    )
    parser.add_argument(
        "--enableEdgeTPU", help="Use EdgeTPU model.", action="store_true"
    )
    args = parser.parse_args()

    run(args.model, args.frameWidth, args.frameHeight,
        args.numThreads, args.enableEdgeTPU)


if __name__ == "__main__":
    main()
