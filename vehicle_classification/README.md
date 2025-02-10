# 🚗 Vehicle Classification
Usage

This project performs comprehensive vehicle detection and recognition using advanced object detection algorithms. The primary features include:

    Object Detection: Identifies various types of vehicles such as buses, cars, trucks, etc.
    Make and Model Recognition: Determines the make and model of the detected vehicles.
    Color Detection: Detects the color of the vehicles.
    Probability Assessment: Provides a probability score indicating the confidence level of the detections.
    Frame Saving: Saves frames from the video where vehicles are detected.
    Vehicle Coordinates: Records the coordinates of each detected vehicle in the frame.

To use this project, ensure that you have the necessary dependencies installed and follow the instructions provided in the setup guide. Run the main script with the appropriate video input, and the system will process the video to detect and recognize vehicles according to the features listed above.


# 1️⃣. Initial Setup
1. **Clone the Repository**: Start by cloning the project repository to your local system 
2. **Navigate to the Project Directory**: After cloning, change into the project directory with:
    ```bash
    cd folder_name
3. **Download the files:
               NotoSansCJK-Black.ttc 
               input.json
               resnet34.pth
               se_resnext50_32x4d-a260b3a4.pth
               yolov8m.pt 
    ```
4. **Run the Analysis Script**: Execute the script :
  
    python vehicle_classification.py
    
