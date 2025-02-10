import ast
import cv2
import numpy as np
import pandas as pd
import os

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

results = pd.read_csv('./test5.csv')

# Load video
video_path = './automatic-number-plate-recognition-python-yolov8-main/sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out89.mp4', fourcc, fps, (width, height))

car_images_dir = './images'
os.makedirs(car_images_dir, exist_ok=True)  # Create the directory if it doesn't exist

saved_cars = set()  # To keep track of saved car IDs

frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            car_id = df_.iloc[row_indx]['car_id']
            if car_id not in saved_cars:
                # Draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                # Save the image of the car with borders
                car_image_filename = os.path.join(car_images_dir, f'car_{car_id}_with_border.jpg')
                cv2.imwrite(car_image_filename, frame)

                # Add the car ID to the set of saved cars
                saved_cars.add(car_id)

            # Draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Crop license plate and add to frame
            # Add your code here to crop and add the license plate to the frame

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

out.release()
cap.release()
