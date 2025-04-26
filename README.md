# smart-parking-system
Smart Parking System using YOLO11 for real-time parking slot detection

A deep learning-based Smart Parking System that detects available and occupied parking spaces using a custom-trained YOLOv11 model.
Trained on the PKLot dataset and deployed on a Raspberry Pi for real-time monitoring.

📋 Table of Contents

1. About the Project

2. Dataset

3. Model Architecture

4. Installation

5. Training

6. Evaluation

7. Deployment on Raspberry Pi

8. Results

9. Future Work

10. License

1. 📖 About the Project
Parking management has increasingly become a critical issue in urban areas with the rise of vehicle ownership.
Manual monitoring systems are inefficient and often lead to wasted time and increased carbon emissions due to vehicles idling while searching for parking spots.

To address this, our project introduces a deep learning-based Smart Parking System capable of automatically detecting whether parking slots are occupied or free using live camera feeds.
The project pipeline includes:

Data acquisition and preparation using the PKLot dataset,

Model training utilizing the Ultralytics YOLO framework,

Validation and performance evaluation,

Deployment on a Raspberry Pi device for real-time inference using a connected camera.

This approach ensures that parking space availability is determined accurately and efficiently without the need for human supervision.

2. 📂 Dataset
The dataset utilized in this project is the widely used PKLot dataset.
Key details include:

Source: Publicly available and accessed through Roboflow.

Composition: Images captured under varying weather conditions and times of day.

Labels: Each image is annotated to reflect whether parking slots are occupied or vacant.

Format: Annotations are structured according to the YOLO format (bounding box coordinates and class labels).

The diversity within the PKLot dataset ensures that the trained model is robust against real-world variations such as lighting changes, different vehicle sizes, and partial occlusions.

3. 🧠 Model Architecture
For the object detection task, we utilized YOLO11, a high-performance version of the YOLO (You Only Look Once) family.
The model is trained using the Ultralytics framework, offering:

Input Resolution: 640x640 pixels

Number of Epochs: 10

Optimization Algorithms: Either SGD or Adam as configured by default

Loss Components:

Localization Loss (for bounding boxes),

Objectness Loss (for confidence score),

Classification Loss (for classifying as occupied or free).

The model’s lightweight nature makes it suitable for real-time inference, even on low-powered devices like Raspberry Pi.

4. ⚙️ Installation
To replicate or build upon this project, follow these steps:

(A). Clone the GitHub Repository:
     git clone https://github.com/your-username/smart-parking-system.git
     cd smart-parking-system
(B). Install the required Python dependencies:
     pip install ultralytics opencv-python matplotlib numpy
(C). If you are using Google Colab, you can install Ultralytics directly:
     !pip install ultralytics
Make sure you have a GPU environment enabled if you wish to accelerate the training and inference process.

5. 🚀 Training the Model
Training involves the following major steps:

(A). Download and prepare the PKLot dataset, ensuring it is properly structured with a data.yaml file defining classes and paths.

(B). Initialize the YOLO model using a pre-trained checkpoint or a scratch model.

(C). Begin training:
     from ultralytics import YOLO

     model = YOLO('yolov8n.pt')  # or your customized YOLO11 model
     model.train(data='path/to/data.yaml', epochs=10, imgsz=640)
The model will learn to predict bounding boxes around parked vehicles and classify parking spots as occupied or free.

6. 📈 Model Evaluation
After training, model performance is evaluated on a separate validation set.

(A). Validation can be triggered easily:
     results = model.val()
(B). You can also visualize predictions on unseen test images:
     model.predict('path/to/test/image.jpg', save=True)
Evaluation metrics like mean Average Precision (mAP), Precision, and Recall are used to assess the model's effectiveness.

7. 🖥️ Deployment on Raspberry Pi
The Smart Parking System is designed to operate efficiently on embedded devices.

Deployment Steps:

Transfer the trained YOLO model (e.g., best.pt) to the Raspberry Pi.

Set up a webcam or Pi camera module.

Execute the following Python script to start live detection:
import cv2
from ultralytics import YOLO

model = YOLO('path/to/best.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.5)
    cv2.imshow('Smart Parking Detection', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

This script captures live frames from the camera, applies YOLO inference, and displays the parking status in real-time.

8. 📊 Experimental Results
After training and testing, the system achieved promising results:
***Check the colab notebook***

9. 🔮 Future Enhancements
Possible improvements and extensions include:

Fine-tuning the model with more training epochs and data augmentation techniques.

Integrating an IoT dashboard to remotely monitor parking lot status.

Employing model quantization and optimization techniques (such as TensorRT or OpenVINO) to further enhance deployment efficiency on edge devices.

Extending to multi-camera, multi-level parking lot monitoring.

10. 📜 License Information
This project is distributed under the MIT License.
Please refer to the LICENSE file for more details.   

