import threading
import cv2
from detection import Camera, utils  # assuming detection.py defines these
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class StopSignDetector:
    """Detects stop signs in a separate thread."""
    def __init__(self, model="efficientdet.tflite", width=640, height=480):
        self.model = model
        self.width = width
        self.height = height
        self.stop_detected = False  # latched detection
        self.handled = False        # whether the car has already stopped
        self.running = True
        self.lock = threading.Lock()

        # Start thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cam = Camera(preview_size=(self.width, self.height))

        base_options = python.BaseOptions(model_asset_path=self.model)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.3,
            max_results=3
        )
        detector = vision.ObjectDetector.create_from_options(options)

        while self.running:
            frame = cam.get_frame()
            if frame is None:
                continue

            # Convert to BGR for Mediapipe Image
            bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            import mediapipe as mp
            input_tensor = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=bgr_image
            )

            result = detector.detect(input_tensor)

            # Check if stop sign detected
            stop_present = any(
                detection.categories[0].category_name.lower() == "stop"
                for detection in result.detections
            )

            with self.lock:
                if stop_present:
                    self.stop_detected = True  # latch it

            # Optional visualization
            vis_image = utils.visualize(bgr_image, result)
            cv2.imshow("Stop Sign Detection", vis_image)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    def get_status(self):
        with self.lock:
            return self.stop_detected

    def shutdown(self):
        self.running = False
        self.thread.join()
