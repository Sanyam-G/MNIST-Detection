# config.py

# Model Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Data Paths
DATA_ROOT = './data'
MODEL_PATH = 'mnist_model.pth'

# Metrics Output Paths
TRAIN_METRICS_CSV = 'train_metrics.csv'
VAL_METRICS_CSV = 'val_metrics.csv'
CONFUSION_MATRIX_CSV = 'confusion_matrix.csv'

# Flask App Settings
FLASK_DEBUG = True
STATIC_VAL_ACC = "0.98" # This should ideally be dynamically updated after training
