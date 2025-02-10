import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'automatic_Number_plate_Recognition')))
from main import run_anpr
from illegally_parking.integrateparking import process_video
from Traffic_Density_Estimation.ex import traffic_video 
from ultralytics import YOLO  # Import YOLO to load models
from vehicle_classification.vehicle_classification.py import load_models, process_video
import torch
from Pedestrian_Detection.detect import  detect_pedestrians
from speed_detection.main import trackMultipleObjects

def main():'
    # Define paths
    video_path = "automatic/sample.mp4"
    output_csv = "automatic/output.csv"
    coco_model_path = "automatic/yolov8n.pt"
    license_plate_detector_path = "automatic/license_plate_detector.pt"

    # Load the models
    coco_model = YOLO(coco_model_path)
    license_plate_detector = YOLO(license_plate_detector_path)
    

    # Call the imported functions
    run_anpr(video_path, output_csv, coco_model, license_plate_detector)
    video_path = "/home/keerthi/project/illegally_parking/video6.mp4"
    yolo_base_path = "/home/keerthi/project/illegally_parking/yolo-coco"
    output_path = "illegally-parking/processed_output.avi"
    process_video(video_path, yolo_base_path, output_path)
    video_path='/home/keerthi/project/Traffic_Density_Estimation/istockphoto-1136516499-640_adpp_is.mp4',
    model_path = "/home/keerthi/project/Traffic_Density_Estimation/models/best.pt"

  #  print(os.path.exists(model_path))



    output_path='output.mp4',
    traffic_video(video_path, model_path,output_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, yolo_model, class_indict = load_models(device)
    video_path = 'vehicle_classification/sample.mp4'
    output_video_path = 'vehicle_classification/output.mp4'
    process_video(video_path, output_video_path, model, yolo_model, class_indict, device)
    input_video_path = "Pedestrian_Detection/ssvid.net - India What pedestrian crossing_v720P.mp4"  # Update with your input video path
    output_video_path = "Pedestrian_Detection/output_video.mp4"  # Update with your output video path
    
    # Call the pedestrian detection function
    detect_pedestrians(input_video_path, output_video_path)
    trackMultipleObjects()


if __name__ == "__main__":
    main()
