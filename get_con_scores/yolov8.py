import cv2
from ultralytics import YOLO
import numpy as np
import csv

# Open the video file
cap = cv2.VideoCapture("/path/to/the/video")

# Load the YOLO model
model = YOLO("yolov8m.pt")

# List to store detection data
detections = []

frame_count = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if we have reached the end of the video
    if not ret:
        break

    frame_count += 1

    # Perform object detection on the frame
    results = model(frame, device="mps")
    result = results[0]

    # Extract object names and confidence scores
    object_names = np.array(result.boxes.cls.cpu(), dtype="int")
    confidence_scores = np.array(result.boxes.conf.cpu(), dtype="float")

    # Store detection data
    for cls, score in zip(object_names, confidence_scores):
        detections.append([frame_count, result.names[cls], float(score)])

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Check for the 'Esc' key (27) to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()

# Write detection data to a CSV file
with open("path/to/csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Object", "Confidence"])
    writer.writerows(detections)
