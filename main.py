import cv2
import torch
import threading
import time
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image

class ObjectDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", confidence_threshold=0.7, skip_frames=5):
        print("Loading model...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        self.skip_frames = skip_frames
        self.current_detections = []
        self.frame = None
        self.stop_thread = False

    def detect_loop(self):
        """Run object detection continuously on the latest frame in a background thread"""
        while not self.stop_thread:
            if self.frame is None:
                time.sleep(0.01)
                continue

            frame = self.frame.copy()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)

            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]

            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                label_name = self.model.config.id2label[label.item()]
                # Ignore person detections
                if label_name.lower() != "person":
                    detections.append({"label": label_name, "score": round(score.item(), 3), "box": box})

            self.current_detections = detections

    def draw_detections(self, frame):
        """Draw detection results on the frame"""
        for detection in self.current_detections:
            x1, y1, x2, y2 = map(int, detection["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection['label']}: {detection['score']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def run_camera(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Camera opened successfully! Press 'q' to quit.")
        print("🚫 Person detections will be ignored.")
        
        # Start detection thread
        thread = threading.Thread(target=self.detect_loop, daemon=True)
        thread.start()

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.skip_frames == 0:
                self.frame = frame  # update frame for background thread

            output_frame = self.draw_detections(frame)
            cv2.imshow("Detecto - Real-Time Object Detection (No Person)", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_thread = True
        thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


def main():
    detector = ObjectDetector(
        model_name="facebook/detr-resnet-50",
        confidence_threshold=0.7,
        skip_frames=5  # Adjust this for performance: higher = smoother but less frequent detection
    )
    detector.run_camera()


if __name__ == "__main__":
    main()