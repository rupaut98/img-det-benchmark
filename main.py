import cv2
from ultralytics import YOLO
import numpy as np

# Open the video file
cap = cv2.VideoCapture("maggie.MOV")

# Load the YOLO model
model = YOLO("yolov8m.pt")

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if we have reached the end of the video
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, device="mps")
    result = results[0]

    # Extract object boundaries and names
    boundaries = np.array(result.boxes.xyxy.cpu(), dtype="int")
    object_names = np.array(result.boxes.cls.cpu(), dtype="int")

    # Draw bounding boxes and labels on the frame
    for cls, bbox in zip(object_names, boundaries):
        (x1, y1, x2, y2) = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, result.names[cls], (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        

    # Display the frame with detected objects
    cv2.imshow("Object Detection", frame)

    # Check for the 'Esc' key (27) to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
