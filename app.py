import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template_string, redirect, url_for
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the same CNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
model = SimpleCNN().to(device)
model_path = "mnist_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
else:
    print("Model file not found. Please run train_model.py first.")

# Define image transformation (MNIST style)
transform_pipeline = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create Flask app
app = Flask(__name__)

# HTML template for the home page
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>MNIST Digit Recognizer</title>
  <style>
    body {
      font-family: Arial, sans-serif; 
      margin: 20px; 
      max-width: 600px;
    }
    h1 { color: #2c3e50; }
    input[type=file], input[type=submit] {
      margin: 5px 0;
    }
  </style>
</head>
<body>
  <h1>MNIST Digit Recognizer</h1>
  <p>Upload an image of a handwritten digit (preferably white digit on dark background)</p>
  <form method="POST" enctype="multipart/form-data" action="{{ url_for('predict') }}">
    <input type="file" name="file">
    <br>
    <input type="submit" value="Upload">
  </form>
  {% if prediction is not none %}
    <h2>Top Prediction: {{ prediction }}</h2>
    <p>Model Performance: Final Validation Accuracy = <strong>{{ val_acc }}</strong></p>
    {% if top3 is not none %}
      <h3>Top 3 Predictions</h3>
      <ol>
        {% for digit, prob in top3 %}
          <li>Digit {{ digit }} (Confidence: {{ (prob * 100)|round(2) }}%)</li>
        {% endfor %}
      </ol>
    {% endif %}
  {% endif %}
</body>
</html>
"""

# For demo purposes, set a static validation accuracy (update with your training results)
STATIC_VAL_ACC = "0.98"  # e.g., 98%


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None, val_acc=STATIC_VAL_ACC)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for('home'))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for('home'))

    # Open the uploaded image
    try:
        img = Image.open(file).convert("L")
    except Exception as e:
        return f"Error processing image: {e}"

    # Optional: invert colors if needed (MNIST digits are white on black)
    img_np = np.array(img)
    if img_np.mean() > 128:
        img = Image.fromarray(255 - img_np)

    # Apply transformation
    img_tensor = transform_pipeline(img).unsqueeze(0).to(device)

    # Get prediction from the model
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        top3_prob = top3_prob.cpu().numpy().flatten()
        top3_catid = top3_catid.cpu().numpy().flatten()

    top3_results = [(int(top3_catid[i]), float(top3_prob[i])) for i in range(3)]
    return render_template_string(HTML_TEMPLATE,
                                  prediction=top3_results[0][0],  # top-1
                                  val_acc=STATIC_VAL_ACC,
                                  top3=top3_results)


if __name__ == "__main__":
    app.run(debug=True)