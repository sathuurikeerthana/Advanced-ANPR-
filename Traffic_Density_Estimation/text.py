import cv2
import numpy as np
from ultralytics import YOLO

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# Load COCO class names
with open('coco.names', 'r') as f:
    class_names = f.read().strip().split('\n')

# Define the threshold for considering traffic as heavy
heavy_traffic_threshold = 5  # Changed to 5 for simultaneous detection

# Define the vertical range for the slice and lane threshold
x1, x2 = 325, 635
lane_threshold = 609

# Define the positions for the text annotations on the image
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

# Define font, scale, and colors for the annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text

# Open the video
cap = cv2.VideoCapture('istockphoto-1136516499-640_adpp_is.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Create a copy of the original frame to modify
        detection_frame = frame.copy()
    
        # Black out the regions outside the specified vertical range
        detection_frame[:x1, :] = 0  # Black out from top to x1
        detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
        
        # Perform inference on the modified frame
        results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
        processed_frame = results[0].plot(line_width=1)
        
        # Restore the original top and bottom parts of the frame
        processed_frame[:x1, :] = frame[:x1, :].copy()
        processed_frame[x2:, :] = frame[x2:, :].copy()        
        
        # Draw horizontal and vertical lines to represent X and Y axes
        height, width = processed_frame.shape[:2]
        cv2.line(processed_frame, (0, x1), (width, x1), (0, 255, 0), 2)  # Horizontal line
        cv2.line(processed_frame, (0, x2), (width, x2), (0, 255, 0), 2)  # Horizontal line
        cv2.line(processed_frame, (lane_threshold, 0), (lane_threshold, height), (255, 0, 0), 2)  # Vertical line
        
        # Retrieve the bounding boxes from the results
        bounding_boxes = results[0].boxes

        # Initialize counters for vehicles in each lane
        vehicles_in_left_lane = 0
        vehicles_in_right_lane = 0

        # Loop through each bounding box to count vehicles in each lane
        for box in bounding_boxes:
            # Extract the bounding box coordinates and the class index
            bbox_data = box.xyxy[0].tolist()
            
            # Ensure the bounding box data has the expected number of values
            if len(bbox_data) == 6:
                x_min, y_min, x_max, y_max, conf, class_idx = bbox_data
            elif len(bbox_data) == 4:
                x_min, y_min, x_max, y_max = bbox_data
                conf = None
                class_idx = None
            else:
                continue  # Skip this bounding box if it doesn't have the expected number of values

            # Check if class_idx is valid
            if class_idx is not None and int(class_idx) < len(class_names):
                class_name = class_names[int(class_idx)]

                # Only count vehicles if the detected class is a vehicle type
                if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                    # Check if the vehicle is in the left lane based on the x-coordinate of the bounding box
                    if x_min < lane_threshold:
                        vehicles_in_left_lane += 1
                    else:
                        vehicles_in_right_lane += 1
                    
                    # Draw the bounding box around the detected vehicle
                    cv2.rectangle(processed_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                else:
                    # If the detected object is not a vehicle, print class name for debugging
                    print(f"Detected non-vehicle object: {class_name}")
            else:
                # Handle cases where class_idx is None or invalid
                print(f"Invalid class index: {class_idx}")

        # Debugging print statements for counts
        print(f"Vehicles in Left Lane: {vehicles_in_left_lane}")
        print(f"Vehicles in Right Lane: {vehicles_in_right_lane}")

        # Determine the traffic intensity for the left lane
        traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
        # Determine the traffic intensity for the right lane
        traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

        # Add a background rectangle for the left lane vehicle count
        cv2.rectangle(processed_frame, (text_position_left_lane[0]-10, text_position_left_lane[1] - 25), 
                      (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)

        # Add the vehicle count text on top of the rectangle for the left lane
        cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the left lane traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_left_lane[0]-10, intensity_position_left_lane[1] - 25), 
                      (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color, -1)

        # Add the traffic intensity text on top of the rectangle for the left lane
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the right lane vehicle count
        cv2.rectangle(processed_frame, (text_position_right_lane[0]-10, text_position_right_lane[1] - 25), 
                      (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)

        # Add the vehicle count text on top of the rectangle for the right lane
        cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the right lane traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_right_lane[0]-10, intensity_position_right_lane[1] - 25), 
                      (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)

        # Add the traffic intensity text on top of the rectangle for the right lane
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Write the processed frame into the output video
        out.write(processed_frame)

        # Display the processed frame
        cv2.imshow('Real-time Traffic Analysis', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
