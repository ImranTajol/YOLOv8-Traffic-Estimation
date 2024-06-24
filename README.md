# Traffic Density Estimation with YOLOv8

## 🔍 Overview
Project aims to estimate the traffic flow by filtering the desired region and apply YOLOv8 algorithm to detect vehicle within the frame. The count of detected cars is tracked to determine the traffic density (High or Low). The threshold for high traffic is hard coded which give flexibility to customize. The selected region for detection is masked using a black and white image which then compute the AND bitwise operation with the original frame.

### 🔍 Specifications 
- 🚗 **Class**: 'Vehicle' including cars, trucks, and buses.
- 📂 **Format**: YOLOv8 annotation format

## 📁 File Descriptions

- **`images/`**: This directory houses the cover images for the project and the sample image utilized within the notebook.
- **`videos/`**: This directory stores the samples videos used to test the coding 
- **`models/`**: Contains the best-performing fine-tuned YOLOv8 model in both `.pt` (PyTorch format) and `.onnx` (Open Neural Network Exchange format) for broad compatibility. The model was trained using online notebook such as Kaggle and Google Colab.
- **`README.md`**: The document you are reading that offers an insightful overview and essential information about the project.
- **`mask.png`**: mask image to select the desired region for image detection 
- **`real_time_traffic_analysis.py`**: The Python script for deploying the YOLOv8 model to estimate traffic density in real-time on a local system.