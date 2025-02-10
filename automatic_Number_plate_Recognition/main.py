import matplotlib
matplotlib.use('Agg')  # Ensure this is the first import

import numpy as np
import cv2
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

def run_anpr(video_path, output_csv, coco_model, license_plate_detector):
    """Run Automatic Number Plate Recognition on the video."""
    results = {}
    mot_tracker = Sort()
    cap = cv2.VideoCapture(video_path)
    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # Run detection
            detections = coco_model(frame)[0]
            detections_ = []

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in [2, 3, 5, 7]:  # Filter for cars, trucks, etc.
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))

            # Run license plate detection
            license_plates = license_plate_detector(frame)[0]

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                if car_id != -1:
                    # Extract and process license plate crop
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

    write_csv(results, output_csv)
    cap.release()
    print(f"ANPR results saved to {output_csv}")

if __name__ == "__main__":
    # Hardcoded paths
    video_path = "automatic/sample.mp4"
    output_csv = "automatic/output.csv"
    coco_model_path = "automatic/yolov8n.pt"
    license_plate_detector_path = r"C:\Users\keerthana\Downloads\Keerthana_Project\automatic_Number_plate_Recognition\license_plate_detector.pt"

    # Load the models
    coco_model = YOLO(coco_model_path)
    license_plate_detector = YOLO(license_plate_detector_path)

    # Run the ANPR process
    run_anpr(video_path, output_csv, coco_model, license_plate_detector)
