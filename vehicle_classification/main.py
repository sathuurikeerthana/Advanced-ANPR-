import os
import json
import time
import numpy as np
import torch
from torchvision import transforms
import cv2 as cv
from torchvision.models import resnet34

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Define COCO classes used by YOLO models
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant',
    'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Define basic colors and their BGR ranges for detection
COLORS = {
    'Red': ([0, 0, 100], [50, 56, 255]),
    'Green': ([0, 100, 0], [50, 255, 50]),
    'Blue': ([100, 0, 0], [255, 50, 50]),
    'Yellow': ([0, 100, 100], [50, 255, 255]),
    'White': ([200, 200, 200], [255, 255, 255]),
    'Black': ([0, 0, 0], [50, 50, 50])
}

def detect_color(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    color_counter = Counter()
    for color_name, (lower, upper) in COLORS.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv.inRange(image, lower, upper)
        color_counter[color_name] += cv.countNonZero(mask)
    return color_counter.most_common(1)[0][0] if color_counter else 'Unknown'

def cvImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):  # Check if OpenCV image type
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("vehicle_classification/NotoSansCJK-Black.ttc", textSize, encoding="utf-8")
    for i, line in enumerate(text.split('\n')):
        draw.text((left, top + i * textSize), line, textColor, font=fontText)
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

def convert_to_input(frame):
    img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Read class_indict
    json_path = r'C:\Users\keerthana\Downloads\Keerthana_Project\vehicle_classification\input.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r", encoding='utf-8') as json_file:
        class_indict = json.load(json_file)

    # Create model
    model = resnet34(num_classes=1778).to(device)

    # Load model weights
    weights_path = "vehicle_classification/se_resnext50_32x4d-a260b3a4.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Initialize YOLO model
    yolo_model = YOLO("vehicle_classification/yolov8m.pt")  # Change to the path of your YOLO model if needed

    video_path = r'C:\Users\keerthana\Downloads\Keerthana_Project\automatic_Number_plate_Recognition\sample.mp4'
    assert os.path.exists(video_path), f"file: '{video_path}' does not exist."

    cap = cv.VideoCapture(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    output_video_path = 'vehicle_classification/outputvideooo.mp4'
    out = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    temp_list = []
    save_dir = 'frames_with_vehicles'
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    frame_count = 0  # To keep track of saved frames

    # Open CSV file for writing
    csv_file_path = 'vehicle_classification/vehicle_details.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['Frame', 'Class', 'Make', 'Model', 'Color', 'Probability', 'Coordinates', 'Frame Path']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            imgshow = frame.copy()

            # Vehicle detection using YOLO
            results = yolo_model(frame)

            detected_vehicle = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = COCO_CLASSES[class_id]
                    score = box.conf[0].item()
                    
                    # Detect the color of the vehicle
                    vehicle_crop = frame[y1:y2, x1:x2]
                    vehicle_color = detect_color(vehicle_crop)
                    
                    # Convert crop to input format
                    img = convert_to_input(vehicle_crop)

                    # Prediction
                    model.eval()
                    t1 = time.time()
                    with torch.no_grad():
                        output = torch.squeeze(model(img.to(device))).cpu()
                        predict = torch.softmax(output, dim=0)
                        predict_cla = torch.argmax(predict).numpy()

                    if str(predict_cla) in class_indict:
                        class_info = class_indict[str(predict_cla)]
                        make = class_info.get('make', 'Unknown')
                        model_name = class_info.get('model', 'Unknown')
                        details = (f"Make: {make}\n"
                                  f"Model: {model_name}\n"
                                  f"Color: {vehicle_color}")
                    else:
                        details = (f"Make: Unknown, Model: Unknown\n"
                                  f"Color: {vehicle_color}")

                    # Save the frame if a vehicle is detected
                    if not detected_vehicle:
                        frame_filename = os.path.join(save_dir, f'frame_{frame_count}.jpg')
                        cv.imwrite(frame_filename, frame)
                        frame_count += 1
                        detected_vehicle = True

                        # Add the file path to details
                        details += f"\nFrame Path: {frame_filename}"

                    # Format print_res with structured details
                    print_res = (f"Class: {class_name}\n"
                                  f"Details: {details}\n"
                                  f"Prob: {predict[predict_cla].numpy():.3f}\n"
                                  f"Coordinates: ({x1}, {y1}, {x2}, {y2})")
                    
                    print(print_res)
                    print('res_time=' + str(time.time() - t1))
                    print("left", x1, "right", y1, "up", x2, "below", y2)

                    # Draw the bounding box
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame = cvImgAddText(frame, print_res, x1, y1 - 50, (255, 255, 0), 20)
                    
                    # Write vehicle details to CSV
                    writer.writerow({
                        'Frame': frame_count,
                        'Class': class_name,
                        'Make': make,
                        'Model': model_name,
                        'Color': vehicle_color,
                        'Probability': f"{predict[predict_cla].numpy():.3f}",
                        'Coordinates': f"({x1}, {y1}, {x2}, {y2})",
                        'Frame Path': frame_filename
                    })

            out.write(frame)
            

    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
