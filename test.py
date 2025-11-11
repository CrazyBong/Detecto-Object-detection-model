import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import numpy as np
import os

def test_image_detection(image_path, confidence_threshold=0.7):
    """
    Test object detection on a single image
    
    Args:
        image_path: Path to test image
        confidence_threshold: Minimum confidence score
    """
    print(f"Testing object detection on: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Load model
    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes, 
        threshold=confidence_threshold
    )[0]
    
    # Convert to OpenCV format for visualization
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Draw detections
    print(f"\nDetected {len(results['scores'])} objects:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        
        print(f"  - {label_name}: {confidence} at {box}")
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        text = f"{label_name}: {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            cv_image, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            cv_image, 
            text, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            2
        )
    
    # Save and display result
    output_path = "test_output.jpg"
    cv2.imwrite(output_path, cv_image)
    print(f"\nResult saved to: {output_path}")
    
    # Display image
    cv2.imshow('Object Detection Result - Press any key to close', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_webcam_snapshot(confidence_threshold=0.7):
    """
    Capture a single frame from webcam and test detection
    
    Args:
        confidence_threshold: Minimum confidence score
    """
    print("Opening webcam for snapshot test...")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press SPACE to capture, ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Show preview
        cv2.imshow('Webcam Preview - SPACE to capture, ESC to cancel', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == 32:  # SPACE
            print("Capturing frame...")
            # Save captured frame
            cv2.imwrite("webcam_capture.jpg", frame)
            cap.release()
            cv2.destroyAllWindows()
            
            # Test detection on captured frame
            test_image_detection("webcam_capture.jpg", confidence_threshold)
            break


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Object Detection Test Suite")
    print("=" * 60)
    print("\nSelect test mode:")
    print("1. Test on existing image")
    print("2. Capture from webcam and test")
    print("3. Quick model verification")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        image_path = input("Enter image path: ").strip()
        threshold = float(input("Enter confidence threshold (0.5-0.9, default 0.7): ") or "0.7")
        test_image_detection(image_path, threshold)
    
    elif choice == "2":
        threshold = float(input("Enter confidence threshold (0.5-0.9, default 0.7): ") or "0.7")
        test_webcam_snapshot(threshold)
    
    elif choice == "3":
        print("\nVerifying model installation...")
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            print("✓ Model loaded successfully!")
            print(f"✓ Model can detect {len(model.config.id2label)} object classes")
            print("\nSample classes:", list(model.config.id2label.values())[:10])
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    run_tests()