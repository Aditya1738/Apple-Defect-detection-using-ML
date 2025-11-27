# Apple Defect Detection using ML
  Apple defect detection and disease classification using CNN models on fruit images.
# Apple Defect Detection using ML

Detects apples in webcam frames and classifies them into healthy or diseased categories using deep learning.

## Features

- Real-time fruit type detection (apple vs other fruits).
- Apple disease classification: blotch, rot, scab, and normal.
- Uses two trained TensorFlow/Keras models (`best_fruit_model.h5`, `cnn_model_final_apple.keras`).
- Webcam-based live prediction with OpenCV.

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## Dataset

This project uses the **"FRUITS DATASET FOR FRUIT DISEASE CLASSIFICATION"** from Kaggle.[web:22]  
Dataset link: https://www.kaggle.com/datasets/ateebnoone/fruits-dataset-for-fruit-disease-classification[web:22]  

Download the dataset from Kaggle and place it in a `data/` folder following your local structure.

## How to Run

1. Create and activate a Python environment.
2. Install dependencies:
3. Ensure the model files are in the project root:
- `best_fruit_model.h5`
- `cnn_model_final_apple.keras`
4. Run the script:
5. A webcam window will open. Press `q` to quit.

## Project Structure

- `main.py` – main script for webcam capture, detection, and classification.
- `best_fruit_model.h5` – model for fruit type detection.
- `cnn_model_final_apple.keras` – model for apple disease classification.
- `data/` – (not in repo) folder where you place the Kaggle dataset locally.

## License

This project is for educational purposes.  
Please check the Kaggle dataset page for its specific license and usage terms.[web:22]

## Here are some ScreenShots of Project 
<img width="864" height="777" alt="Screenshot 2025-08-29 121646" src="https://github.com/user-attachments/assets/0a11575d-3c74-4e4c-b74d-f43ec68983c4" />
<img width="929" height="822" alt="Screenshot 2025-08-29 121442" src="https://github.com/user-attachments/assets/9e09299b-dcd8-4e7b-9507-41fd5e983b24" />
<img width="591" height="518" alt="Screenshot 2025-07-13 120751" src="https://github.com/user-attachments/assets/14aff1a2-a840-4def-a230-2d96bff6615c" />

