# AI-Based PCB Quality Inspection

## 📌 Project Overview

AI-Based PCB Quality Inspection is a computer vision project designed to detect and classify defects in Printed Circuit Boards (PCBs) using Artificial Intelligence and Deep Learning.

The system uses a **YOLO11n object detection model** trained to identify six types of PCB defects. OpenCV supports image processing and camera input, Streamlit provides a user-friendly web interface, and SQLite stores inspection records for future reference.

The system supports PCB inspection through uploaded images and a USB camera.

## 🎯 Objectives

- Automatically detect and classify PCB defects using AI.
- Reduce manual inspection effort and human error.
- Support inspection of uploaded PCB images and camera input.
- Display detected defects with bounding boxes and confidence scores.
- Automatically store inspection results, including defect-free inspections.
- Maintain inspection history using an SQLite database.
- Provide a simple and interactive dashboard for PCB quality monitoring.

## ✨ Features

- **AI-Based Detection:** Uses a trained YOLO11n model for PCB defect detection.
- **Six Defect Classes:** Detects six different types of PCB defects.
- **Image Upload:** Inspect PCB images uploaded by the user.
- **USB Camera Support:** Capture PCB images using a connected camera.
- **Automatic Inspection History:** Store inspection results in SQLite.
- **Defect-Free Recording:** Save records even when no defects are detected.
- **Visual Results:** Display bounding boxes and detection confidence scores.
- **Inspection Dashboard:** View previously recorded inspection results through Streamlit.

## 🔍 PCB Defect Classes

The trained model identifies the following six PCB defects:

| No. | Defect Class | Description |
|---|---|---|
| 1 | Mouse Bite | Small irregular cuts or notches along the PCB edge. |
| 2 | Spur | Unwanted copper projections extending from a copper trace. |
| 3 | Missing Hole | A required hole is absent from its intended position. |
| 4 | Short | Unwanted electrical connection between conductive paths. |
| 5 | Open Circuit | A break in a conductive path that interrupts electrical continuity. |
| 6 | Spurious Copper | Unwanted copper material present in an unintended area. |

## 🛠️ Technologies Used

- **Python** – Main programming language
- **YOLO11n** – PCB defect detection model
- **Ultralytics** – Model training and inference
- **OpenCV** – Image processing and camera capture
- **Streamlit** – Web application and dashboard
- **SQLite** – Inspection history database
- **Pillow (PIL)** – Image handling
- **NumPy** – Numerical and image data operations
- **Pandas** – Inspection data handling

## ⚙️ System Workflow

1. The user uploads a PCB image or uses a USB camera.
2. The input image is processed and passed to the trained YOLO11n model.
3. The model detects and classifies PCB defects.
4. The system displays the inspection results and detected defect locations.
5. The inspection result is automatically saved in the SQLite database.
6. The inspection dashboard allows the user to review previous records.

For camera-based inspection, the system can periodically inspect the PCB and save the results automatically, depending on the configured inspection interval.

## 📂 Project Structure

```text
AI-Based-PCB-Quality-Inspection/
│
├── app.py                  # Streamlit web application
├── best.pt                 # Trained YOLO11n model
├── camera_test.py          # USB camera testing script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── inspection_images/      # Saved inspection images
├── test_images/            # Images used for testing
└── runs/                   # Model inference outputs
```

The SQLite database file may also be created by the application's database module, depending on its configuration.

## 💻 System Requirements

- Windows 10 or Windows 11
- Python 3.11
- Minimum 8 GB RAM recommended
- Webcam or USB camera for camera-based inspection
- Sufficient storage for the trained model, images, and inspection database

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project folder:

```bash
cd AI-Based-PCB-Quality-Inspection
```

Alternatively, open your existing project folder in Anaconda Prompt.

### 2. Create a Conda Environment

```bash
conda create -n pcb_inspection python=3.11
```

Activate the environment:

```bash
conda activate pcb_inspection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the Trained Model

Ensure that `best.pt` is present in the project folder.

You can verify that the model loads successfully by running:

```bash
python -c "from ultralytics import YOLO; m=YOLO('best.pt'); print(m.names)"
```

The model should display the six trained defect classes.

### 5. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 📊 Model Information

| Parameter | Details |
|---|---|
| Model | YOLO11n |
| Task | Object detection |
| Number of classes | 6 |
| Input image size | 640 × 640 during the configured training and inference |
| Training epochs | 50 configured |
| Model weights | `best.pt` |
| Dataset | PCB Defect Dataset from Kaggle |

The model was trained to identify six PCB defect categories. Actual detection performance depends on image quality, lighting, PCB appearance, and the training data.

## 🗄️ Database

SQLite is used to maintain inspection history.

The database can store details such as:

- Inspection ID
- Inspection date and time
- Inspection result
- Detected defect names
- Defect count
- Image or result information, depending on the implemented database schema

Both defective and defect-free PCB inspections can be recorded automatically when the application is configured for automatic history saving.

## 🧪 Testing

The system can be tested using:

1. PCB images containing known defects.
2. PCB images without visible defects.
3. USB camera input.
4. Different PCB orientations and lighting conditions.
5. Inspection history to verify that records are stored correctly.

Compare the model's predictions with the actual defects to evaluate its performance.

## 🔮 Future Enhancements

- Integrate a high-resolution industrial camera.
- Improve detection accuracy using additional training data.
- Add inspection reports and downloadable summaries.
