
import cv2
import numpy as np
from ultralytics import YOLO
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
def traffic_video(video_path, model_path, output_path, heavy_traffic_threshold=5):
    # Load the best fine-tuned YOLOv8 model
    best_model = YOLO(model_path)

    # Define the threshold for considering traffic as heavy
    heavy_traffic_threshold = heavy_traffic_threshold

    # Define the vertices for the quadrilaterals
    vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
    vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

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
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Initialize counters and trackers
    vehicles_crossed = 0
    tracked_vehicles = []

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Create a copy of the original frame to modify
            detection_frame = frame.copy()
        
            # Get the frame dimensions
            height, width, _ = frame.shape
            
            # Calculate the position of the X-axis (middle of the frame)
            x_axis_y = height // 2

            # Black out the regions outside the specified vertical range
            detection_frame[:x1, :] = 0  # Black out from top to x1
            detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
            
            # Perform inference on the modified frame
            results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
            processed_frame = results[0].plot(line_width=1)
            
            # Restore the original top and bottom parts of the frame
            processed_frame[:x1, :] = frame[:x1, :].copy()
            processed_frame[x2:, :] = frame[x2:, :].copy()        
            
            # Draw the quadrilaterals on the processed frame
            cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

            # Draw the X-axis line in the middle of the frame
            cv2.line(processed_frame, (0, x_axis_y), (frame.shape[1], x_axis_y), (0, 255, 255), 2)  # Yellow line

            # Retrieve the bounding boxes from the results
            bounding_boxes = results[0].boxes
            vehicle_count_left_lane = 0
            vehicle_count_right_lane = 0

            # Loop through each bounding box to count vehicles in each lane and detect crossing
            for box in bounding_boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                # Draw bounding box around the detected vehicle
                cv2.rectangle(processed_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                # Determine if the vehicle is in the left or right lane based on its position
                if x_max < lane_threshold:
                    vehicle_count_left_lane += 1
                elif x_min > lane_threshold:
                    vehicle_count_right_lane += 1

                # Check if the vehicle crosses the X-axis
                vehicle_center_y = (y_min + y_max) / 2
                if y_min < x_axis_y < y_max:
                    # Check if the vehicle has crossed the X-axis
                    if (box.id not in tracked_vehicles):
                        vehicles_crossed += 1
                        tracked_vehicles.append(box.id)
            
            # Determine the traffic intensity for each lane
            traffic_intensity_left = "Heavy" if vehicle_count_left_lane > heavy_traffic_threshold else "Smooth"
            traffic_intensity_right = "Heavy" if vehicle_count_right_lane > heavy_traffic_threshold else "Smooth"

            # Add a background rectangle for the left lane vehicle count
            cv2.rectangle(processed_frame, (text_position_left_lane[0]-10, text_position_left_lane[1] - 25), 
                          (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)
            # Add the vehicle count text on top of the rectangle for the left lane
            cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicle_count_left_lane}', text_position_left_lane, 
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
            cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicle_count_right_lane}', text_position_right_lane, 
                        font, font_scale, font_color, 2, cv2.LINE_AA)

            # Add a background rectangle for the right lane traffic intensity
            cv2.rectangle(processed_frame, (intensity_position_right_lane[0]-10, intensity_position_right_lane[1] - 25), 
                          (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)
            # Add the traffic intensity text on top of the rectangle for the right lane
            cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane, 
                        font, font_scale, font_color, 2, cv2.LINE_AA)

            # Add a background rectangle for the number of vehicles crossed the X-axis
            cv2.rectangle(processed_frame, (text_position_right_lane[0]-10, text_position_right_lane[1] + 50), 
                          (text_position_right_lane[0] + 460, text_position_right_lane[1] + 80), background_color, -1)
            # Add the number of vehicles crossed the X-axis text on top of the rectangle
            cv2.putText(processed_frame, f'Vehicles Crossed X-Axis: {vehicles_crossed}', (text_position_right_lane[0], text_position_right_lane[1] + 75), 
                        font, font_scale, font_color, 2, cv2.LINE_AA)

            # Display the processed frame
            cv2.imshow('Real-time Traffic Analysis', processed_frame)

            # Write the processed frame into the output video
            out.write(processed_frame)

            # Press Q on keyboard to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture and video write objects
    cap.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()

# Call the function with the specified inputs
traffic_video(
    video_path='Traffic_Density_Estimation/istockphoto-1136516499-640_adpp_is.mp4',
    model_path='Traffic_Density_Estimation/models/best.pt',
    output_path='output.mp4',
    heavy_traffic_threshold=5
)