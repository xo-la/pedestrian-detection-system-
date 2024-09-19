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

### 2. Setup Python Virtual Environment

To isolate the dependencies for this project, we recommend using a virtual environment.

```bash
python -m venv venv
source env/Scripts/activate
```

It will create a light weight virtual environment and activate it.

### 3. Install Dependencies

Ensure you have Python installed. Run the following command to install the necessary Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Next, download the `YOLOv5` code from it's official repository:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 4. Running the Project

Once the dependencies are installed, you're ready to run the pedestrian detection system. To run it, you can use the following steps:

1. **Capture Video from Webcam:** By default, the script captures video from your webcam.

2. **Detect Pedestrians:** YOLOv5 will process each frame and detect pedestrians.

3. **View Results:** Bounding boxes will appear around detected pedestrians in the real-time video feed.

### 5. How to run the script

Navigate back to the main directory and run the python script:

```bash
cd ../ 
python <python script>
```

**Note:** Replace the webcam input (0) with the path to a video file if you want to process a pre-recorded video.

## 💡 Project Demo

Here's a quick guide on how the project works step-by-step:

1. **Video Input:** The script captures video input either from a webcam or a video file.

2. **YOLOv5 Model:** The YOLOv5 pre-trained model is loaded using PyTorch. The model is trained on the COCO dataset which includes pedestrians.

3. **Pedestrian Detection:** The system processes each frame from the video feed and detects pedestrians.

4. **Real-Time Display:** The detections are displayed in real-time, with bounding boxes drawn around the pedestrians.

## 📜 YOLOv5 Pedestrian Detection Code Overview

### YOLOv5 Model

- The YOLOv5 model is loaded using torch.hub, with the pre-trained weights from the COCO dataset.

- The model can detect multiple objects, but we focus on the class person, which is indexed as 0 in the COCO dataset.

### Detection Process

- The function detect_pedestrians() processes each frame to detect pedestrians.

- It resizes the frame to 640x640, feeds it to YOLOv5, and draws bounding boxes around detected individuals.

### Video Processing

- The script uses cv2.VideoCapture(0) to capture video input from a webcam. Change this to a file path to use a pre-recorded video.

- The cv2_imshow function from Google Colab is used to display the real-time detection in a window.

## 🔧 Customize the Project

You can modify the script to:

- **Use a different input video:** Replace the webcam input (0) with a video file path.

- **Process a pre-recorded video:** Edit the cv2.VideoCapture line to use a video file.

- **Tweak the detection threshold:** Adjust the confidence score to fine-tune pedestrian detection.

## 🤝 Contributing

We welcome contributions to make this project even better! Whether you want to add new features, improve the documentation, or optimize the detection system, feel free to fork the repository and submit a pull request.

### Steps to Contribute

1. Fork the repository.

2. Create a new branch for your feature

3. Make your changes and test thoroughly.

4. Submit a pull request with a description of your changes.

## ⭐ Show Your Support

If you find this project helpful, give us a ⭐ on GitHub! Your support will help improve this project and make it better for the community.

**"Detect better, drive safer!"** 🚗🔍✨
