import math

import cv2
from ultralytics import YOLO


def check_camera_access():
    """Check if camera is accessible and return appropriate camera index."""
    for index in range(5):  # Check cameras 0-4
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera found at index {index}")
                cap.release()
                return index
        cap.release()
    return None


# Find available camera
camera_index = check_camera_access()
if camera_index is None:
    print("No camera found. Please check camera permissions and connections.")
    exit()

# Start webcam with found camera index
cap = cv2.VideoCapture(camera_index)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Load YOLO model
try:
    model = YOLO("yolo-Weights/yolov8n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Make sure the model file exists in the yolo-Weights directory")
    cap.release()
    exit()

# COCO class names
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

print("Starting object detection. Press 'q' to quit.")

# Main loop with proper error handling
while cap.isOpened():
    success, img = cap.read()

    if not success or img is None:
        print("Failed to read from camera")
        break

    # Check if image is valid
    if img.size == 0:
        print("Empty image received")
        continue

    try:
        # Run YOLO inference
        results = model(img, stream=True, verbose=False)

        # Process detections
        for r in results:
            boxes = r.boxes

            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Get confidence score
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # Get class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    # Create label text
                    label = f"{class_name} {confidence:.2f}"

                    # Add text background for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        img,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        (255, 0, 255),
                        -1,
                    )

                    # Draw label text
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

        # Display the frame
        cv2.imshow("YOLO Object Detection", img)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error during processing: {e}")
        continue

# Clean up
cap.release()
cv2.destroyAllWindows()
