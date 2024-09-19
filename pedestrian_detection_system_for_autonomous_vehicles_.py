# Import necessary libraries
import cv2  # OpenCV for video processing
import torch  # PyTorch to run YOLOv5 model
import numpy as np  # To handle array manipulations


# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Classes in the COCO dataset, 'person' is at index 0
COCO_CLASSES = model.names  # ['person', 'bicycle', 'car', ...]

# Initialize video capture from a webcam
cap = cv2.VideoCapture(0)

# Function to process each frame and detect pedestrians
def detect_pedestrians(frame):
    """
    Detect pedestrians in the input video frame.
    Args:
        frame (ndarray): Input video frame from the webcam or video.
    Returns:
        frame (ndarray): Frame with bounding boxes and labels drawn.
    """
    # Resize the frame to 640x640 (size expected by YOLOv5)
    resized_frame = cv2.resize(frame, (640, 640))

    # Perform detection using YOLOv5
    results = model(resized_frame)

    # Extract the detection results
    detections = results.xyxy[0].numpy()

    # Loop through detections and draw bounding boxes for pedestrians
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # If the detected class is 'person' (class_id 0)
        if int(class_id) == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Start processing the video feed
while cap.isOpened():
    ret, frame = cap.read()

    # Check if frame is successfully captured
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Detect pedestrians in the frame
    output_frame = detect_pedestrians(frame)

    # Display the output frame with bounding boxes
    cv2.imshow("Pedestrian Detection", output_frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
