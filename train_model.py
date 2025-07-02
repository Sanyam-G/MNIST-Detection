import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
from model import SimpleCNN, train_transform, test_transform
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = config.BATCH_SIZE
learning_rate = config.LEARNING_RATE
num_epochs = config.NUM_EPOCHS

# MNIST dataset and data loaders
train_dataset = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=True,
                                           download=True, transform=train_transform)
test_dataset  = torchvision.datasets.MNIST(root=config.DATA_ROOT, train=False,
                                           download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training and validation metrics
training_metrics = []
validation_metrics = []
all_preds = []
all_labels = []

# Training and evaluation loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= total_train
    train_accuracy = train_correct / total_train

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    total_val = 0
    # Clear previous epoch's predictions and labels for confusion matrix
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= total_val
    val_accuracy = val_correct / total_val

    training_metrics.append({"epoch": epoch + 1, "loss": train_loss, "accuracy": train_accuracy})
    validation_metrics.append({"epoch": epoch + 1, "loss": val_loss, "accuracy": val_accuracy})
    logging.info(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), config.MODEL_PATH)
logging.info(f"Training complete. Final validation accuracy: {val_accuracy:.4f}")

# Save metrics to CSV files
pd.DataFrame(training_metrics).to_csv(config.TRAIN_METRICS_CSV, index=False)
pd.DataFrame(validation_metrics).to_csv(config.VAL_METRICS_CSV, index=False)

# Compute and save confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds)
conf_mat_df = pd.DataFrame(conf_mat,
                           index=[f"True {i}" for i in range(10)],
                           columns=[f"Pred {i}" for i in range(10)])
conf_mat_df.to_csv(config.CONFUSION_MATRIX_CSV)
logging.info("Training complete. Metrics saved to train_metrics.csv, val_metrics.csv, and confusion_matrix.csv.")