# Handwritten Digit Recognition Web App

This project is a deployable web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) built with PyTorch and served with Flask. The model is trained on the MNIST dataset and achieves approximately 98% validation accuracy.

## Features

- **Model Training:** Trains a CNN on MNIST using PyTorch.
- **Image Preprocessing:** Applies resizing, color inversion, and data augmentation techniques to improve model robustness.
- **Web Interface:** Provides a Flask-based web app where users can upload an image of a handwritten digit and receive a prediction.
- **Top-3 Predictions:** Displays the top three predicted digits along with their confidence scores.
- **Live Demo:** Hosted on PythonAnywhere for public access.

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   
2. **Install Dependencies:** Ensure you have Python 3 installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   (The `requirements.txt` file should include dependencies such as Flask, torch, torchvision, Pillow, opencv-python, etc.)

3. Run the Flask App: Start the web server by running:
   ```bash
   python app.py
   ```
   Then open your browser at [http://127.0.0.1:5000]

## Live Demo
A live version is hosted on PythonAnywhere: [Live Demo Link](https://sammy56656.pythonanywhere.com/)

## Project Structure
The project directory should have the following structure:
```bash
├── app.py               # Flask web application
├── train_model.py       # Model training script
├── mnist_model.pth      # Trained model weights (generated)
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
```
