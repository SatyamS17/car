import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from picamera2 import Picamera2
from libcamera import Transform
import utils


class StopSign:
    def __init__(self, model="efficientdet.tflite", width=640, height=480,
                 num_threads=4, enable_edgetpu=False,
                 hflip=False, vflip=False, headless=False):
        """Initialize camera and detector."""

        self.model = model
        self.width = width
        self.height = height
        self.num_threads = num_threads
        self.enable_edgetpu = enable_edgetpu
        self.headless = headless
        self.stop_detected = False

        # camera setup
        self.camera = Picamera2()
        self.transform = Transform(hflip=1 if hflip else 0,
                                   vflip=1 if vflip else 0)
        preview_config = self.camera.create_preview_configuration(
            main={"size": (self.width, self.height)},
            transform=self.transform
        )
        self.camera.configure(preview_config)
        self.camera.start()

        # setup detector once
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                               score_threshold=0.3,
                                               max_results=3)
        self.detector = vision.ObjectDetector.create_from_options(options)

    def capture_and_detect(self):
        """Capture one frame and run detection."""
        image = self.camera.capture_array()
        if image is None:
            print("ERROR: Unable to read from PiCamera2.")
            return None

        # Convert to OpenCV BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Wrap for mediapipe
        input_tensor = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=bgr_image
        )

        # Run detection
        detection_result = self.detector.detect(input_tensor)

        # Draw results
        for detection in detection_result.detections:
            if detection.categories[0].category_name == "stop sign":
                self.stop_detected = True

        if not self.headless:
            cv2.imshow("object_detector", bgr_image)
            if cv2.waitKey(1) == 27:  # ESC
                return "quit"
        else:
            cv2.imwrite("last_frame.jpg", bgr_image)

        return detection_result

    def main(self):
        """Run the detection loop once per second until stopped."""
        try:
            while True:
                result = self.capture_and_detect()
                if result == "quit":
                    break
                print("Detection result:", result)
                time.sleep(1)  # wait 1 second
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    stop_sign = StopSign(headless=True)  # change to False for GUI
    stop_sign.main()
