import numpy as np
import cv2
import os
import time

def process_video(video_path, output_path, confidence_threshold=0.5, nms_threshold=0.3):
    """Process video for speed detection and save the results."""
    yolo_base_path = "illegally-parking/yolo-coco"
    labelsPath = os.path.sep.join([yolo_base_path, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    weightsPath = os.path.sep.join([yolo_base_path, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_base_path, "yolov3.cfg"])

    print("Loading YOLO...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (H, W) = frame.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        print("Time taken for detection {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []
        temp = 0

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        t = 0
        if len(idxs) > 0:
            for i in idxs.flatten():
                if LABELS[classIDs[i]] == "car":
                    t = 1
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    temp = temp + 1

        if t == 1:
            print("Vehicle detected")
        else:
            print("No Vehicle found")

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")
