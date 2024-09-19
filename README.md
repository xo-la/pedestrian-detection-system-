# 🚶‍♂️ Pedestrian Detection System for Autonomous Vehicles 🚗🤖

This project leverages the **YOLOv5** model for real-time pedestrian detection in video feeds, making it a valuable tool for autonomous vehicles or surveillance systems. YOLOv5 (You Only Look Once) is a state-of-the-art object detection model known for its speed and accuracy. The system detects pedestrians in real-time, drawing bounding boxes around detected individuals in the video feed.

## 🧠 Features

- **Real-Time Detection**: Detect pedestrians in real-time using YOLOv5.

- **Pre-Trained Model**: Utilizes the pre-trained YOLOv5 model on the COCO dataset.

- **High Accuracy**: YOLOv5 is fast and accurate, making it suitable for real-time applications.

- **Customization**: Can be adapted to different video inputs such as webcams or video files.

---

## 🛠️ Project Setup

Follow these steps to clone, install dependencies, and run the project.

### 1. Clone the Repository

```bash
git clone https://github.com/xo-la/pedestrian-detection-system-.git
cd pedestrian-detection-system-
```

### 2. Install Dependencies

Ensure you have Python installed. Run the following commands to install the necessary Python libraries.

```bash
pip install torch torchvision opencv-python pillow
```

Next, download the `YOLOv5` code from it's official repository:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 3. Running the Project

Once the dependencies are installed, you're ready to run the pedestrian detection system. To run it, you can use the following steps:

1. **Capture Video from Webcam:** By default, the script captures video from your webcam.

2. **Detect Pedestrians:** YOLOv5 will process each frame and detect pedestrians.

3. **View Results:** Bounding boxes will appear around detected pedestrians in the real-time video feed.
