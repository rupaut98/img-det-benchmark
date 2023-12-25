import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Open the video file
cap = cv2.VideoCapture("maggie.MOV")

# Load the YOLO model
model = YOLO("yolov5m6.pt")

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

    # Create a DataFrame for the results of this frame
    frame_data = {
        'Frame': frame_number,
        'Object Name': [result.names[cls] for cls in object_names],
        'X1': [bbox[0] for bbox in boundaries],
        'Y1': [bbox[1] for bbox in boundaries],
        'X2': [bbox[2] for bbox in boundaries],
        'Y2': [bbox[3] for bbox in boundaries]
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
full_results_df.to_csv('/Users/rupakraut/Documents/img-det-benchmark/object_detection_results-v5.csv', index=False)
