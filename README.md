# Handwritten Digit Recognition Web App

This project is a deployable web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) built with PyTorch and served with Flask. The model is trained on the MNIST dataset and achieves approximately 98% validation accuracy.

## Features

- **Model Training:** Trains a CNN on MNIST using PyTorch, with metrics and confusion matrix generation.
- **Image Preprocessing:** Applies resizing, color inversion, and data augmentation techniques to improve model robustness.
- **Interactive Web Interface:** Provides a Flask-based web app where users can draw a digit on a canvas for **live prediction** or upload an image of a handwritten digit. The UI/UX has been refined with a clean, minimalist design using a pale color palette and subtle styling.
- **Top-3 Predictions:** Displays the top three predicted digits along with their confidence scores.
- **Configuration Management:** Utilizes a `config.py` file for easy management of hyperparameters and application settings.

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone (https://github.com/Sanyam-G/MNIST-Detection)
   cd handwritten-digit-recognition
   ```
   
2. **Install Dependencies:** Ensure you have Python 3 installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   (The `requirements.txt` file is generated using `pip freeze` and includes all necessary dependencies.)

3. **Train the Model (Optional but Recommended):**
   ```bash
   python train_model.py
   ```
   This will train the CNN model and save the weights to `mnist_model.pth`, along with training/validation metrics and a confusion matrix in CSV format.

4. **Run the Flask App (Local):** Start the web server by running:
   ```bash
   python flask_app.py
   ```
   Then open your browser at [http://127.0.0.1:5000] (or the address shown in the console).

## Dockerization

To run the application using Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t mnist-digit-recognizer .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 5000:5000 mnist-digit-recognizer
    ```
    The application will be accessible in your browser at `http://localhost:5000`.

## Project Structure
The project directory should have the following structure:
```bash
├── flask_app.py         # Flask web application
├── train_model.py       # Model training script
├── model.py             # Defines the CNN model architecture and transformations
├── config.py            # Configuration for hyperparameters and application settings
├── mnist_model.pth      # Trained model weights (generated by train_model.py)
├── requirements.txt     # Python dependencies
├── README.md            # Project overview
├── Dockerfile           # Dockerfile for containerization
├── .dockerignore        # Specifies files to ignore when building Docker image
├── confusion_matrix.csv # Generated by train_model.py
├── train_metrics.csv    # Generated by train_model.py
├── val_metrics.csv      # Generated by train_model.py
├── templates/           # HTML templates for the Flask app
│   └── index.html
└── static/              # Static assets for the Flask app
    ├── css/
    │   └── style.css
    └── js/
        └── canvas.js
```

## Future Improvements

- **Hyperparameter Tuning:** Implement a framework (e.g., Optuna, Weights & Biases) for systematic hyperparameter optimization.
- **Experiment Tracking:** Integrate tools like MLflow or TensorBoard to log and visualize training runs, metrics, and model artifacts for better reproducibility and analysis.
- **Model Checkpointing & Early Stopping:** Save the best model during training and implement early stopping to prevent overfitting.
- **API Endpoint:** Add a dedicated REST API endpoint for model inference, allowing other applications to easily integrate with the model.
- **Unit Tests:** Write comprehensive unit tests for the model, data transformations, and Flask application endpoints to ensure code quality and reliability.
