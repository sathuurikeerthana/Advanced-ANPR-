import cv2
import os
import dlib
import time
from datetime import datetime
import numpy as np

# Disable display GUI for environments without a display (like servers)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Classifier for detecting cars
carCascade = cv2.CascadeClassifier('/home/keerthi/vehicle-speed-detection-master/files/HaarCascadeClassifier.xml')

# Take video
video = cv2.VideoCapture('/home/keerthi/vehicle-speed-detection-master/files/sample.mp4')

WIDTH = 1280  # Width of video frame
HEIGHT = 720  # Height of video frame
cropBegin = 240  # Crop video frame from this point
mark1 = 120  # Mark to start timer
mark2 = 360  # Mark to end timer
markGap = 15  # Distance in meters between the markers
fpsFactor = 3  # To compensate for slow processing
speedLimit = 20  # Speed limit in Kmph
startTracker = {}  # Store starting time of cars
endTracker = {}  # Store ending time of cars

# Make directory to store over-speeding car images
if not os.path.exists('overspeeding/samplecars/'):
    os.makedirs('overspeeding/samplecars/')

print('Speed Limit Set at 20 Kmph')


def blackout(image):
    xBlack = 360
    yBlack = 300
    triangle_cnt = np.array([[0, 0], [xBlack, 0], [0, yBlack]])
    triangle_cnt2 = np.array([[WIDTH, 0], [WIDTH - xBlack, 0], [WIDTH, yBlack]])
    cv2.drawContours(image, [triangle_cnt], 0, (0, 0, 0), -1)
    cv2.drawContours(image, [triangle_cnt2], 0, (0, 0, 0), -1)
    return image


def saveCar(speed, image):
    now = datetime.today().now()
    nameCurTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")
    link = f'overspeeding/samplecars/{nameCurTime}.jpeg'
    cv2.imwrite(link, image)


def estimateSpeed(carID):
    timeDiff = endTracker[carID] - startTracker[carID]
    speed = round(markGap / timeDiff * fpsFactor * 3.6, 2)
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    carTracker = {}

    while True:
        rc, image = video.read()
        if image is None:
            break

        frameTime = time.time()
        image = cv2.resize(image, (WIDTH, HEIGHT))[cropBegin:720, 0:1280]
        resultImage = blackout(image)
        cv2.line(resultImage, (0, mark1), (1280, mark1), (0, 0, 255), 2)
        cv2.line(resultImage, (0, mark2), (1280, mark2), (0, 0, 255), 2)

        frameCounter += 1

        # Delete car IDs not in frame
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)

        if frameCounter % 60 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                xbar = x + 0.5 * w
                ybar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    tx = int(trackedPosition.left())
                    ty = int(trackedPosition.top())
                    tw = int(trackedPosition.width())
                    th = int(trackedPosition.height())

                    txbar = tx + 0.5 * tw
                    tybar = ty + 0.5 * th

                    if (tx <= xbar <= (tx + tw)) and (ty <= ybar <= (ty + th)) and (x <= txbar <= (x + w)) and (y <= tybar <= (y + h)):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            tx = int(trackedPosition.left())
            ty = int(trackedPosition.top())
            tw = int(trackedPosition.width())
            th = int(trackedPosition.height())

            # Put bounding boxes
            cv2.rectangle(resultImage, (tx, ty), (tx + tw, ty + th), rectangleColor, 2)
            cv2.putText(resultImage, str(carID), (tx, ty - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

            # Estimate speed
            if carID not in startTracker and mark2 > ty + th > mark1 and ty < mark1:
                startTracker[carID] = frameTime

            elif carID in startTracker and carID not in endTracker and mark2 < ty + th:
                endTracker[carID] = frameTime
                speed = estimateSpeed(carID)
                if speed > speedLimit:
                    print(f'CAR-ID : {carID} : {speed} kmph - OVERSPEED')
                    saveCar(speed, image[ty:ty + th, tx:tx + tw])
                else:
                    print(f'CAR-ID : {carID} : {speed} kmph')

        # Display each frame (Optional, depending on your environment)
        # Comment out this line if running on a headless server or no display environment
        cv2.imshow('result', resultImage)
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
