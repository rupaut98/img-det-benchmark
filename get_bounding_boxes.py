import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Open the video file
cap = cv2.VideoCapture("new_york_1.mov")

# Load the YOLO model
model = YOLO("yolov8m.pt")

# Initialize a list to store object detection results for each frame
detection_results = []

frame_number = 0  # Initialize the frame number

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

    # Convert bounding box coordinates to a single array
    bbox_array = [bbox.tolist() for bbox in boundaries]

    # Create a DataFrame for the results of this frame
    frame_data = {
        'Frame': frame_number,
        'Object Name': [result.names[cls] for cls in object_names],
        'BoundingBox': bbox_array
    }

    frame_df = pd.DataFrame(frame_data)
    detection_results.append(frame_df)

    # Display the frame with detected objects
    cv2.imshow("Object Detection", frame)

    # Check for the 'Esc' key (27) to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

    frame_number += 1  # Increment the frame number

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()

# Concatenate the individual DataFrames into one DataFrame
full_results_df = pd.concat(detection_results, ignore_index=True)

# Save the DataFrame to a CSV file with an absolute path
full_results_df.to_csv('/Users/rupakraut/Documents/ML_project/object_detection_results.csv', index=False)
