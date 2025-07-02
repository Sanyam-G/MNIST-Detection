import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, redirect, url_for, flash, get_flashed_messages, jsonify
from PIL import Image
import numpy as np
from model import SimpleCNN, flask_transform_pipeline
import config
import logging
import io
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = SimpleCNN().to(device)
model_path = config.MODEL_PATH
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("Model loaded successfully.")
else:
    logging.error("Model file not found. Please run train_model.py first.")

# Create Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key' # Replace with a strong secret key in production



# For demo purposes, set a static validation accuracy (update with your training results)
STATIC_VAL_ACC = config.STATIC_VAL_ACC  # e.g., 98%


@app.route("/", methods=["GET"])
def home():
    return render_template('index.html', prediction=None, val_acc=STATIC_VAL_ACC)


@app.route("/predict", methods=["POST"])
def predict():
    img = None
    if "image_data" in request.form:
        image_data = request.form["image_data"]
        # Remove the "data:image/png;base64," prefix
        base64_data = image_data.split(',')[1]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(base64_data))).convert("L")
        except Exception as e:
            flash(f"Error processing canvas image: {e}")
            return redirect(url_for('home'))
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            flash('No selected file')
            return redirect(url_for('home'))
        try:
            img = Image.open(file).convert("L")
        except Exception as e:
            flash(f"Error processing uploaded image: {e}")
            return redirect(url_for('home'))
    else:
        flash('No image data or file provided')
        return redirect(url_for('home'))

    if img is None:
        flash('Failed to load image.')
        return redirect(url_for('home'))

    # Optional: invert colors if needed (MNIST digits are white on black)
    img_np = np.array(img)
    if img_np.mean() > 128:
        img = Image.fromarray(255 - img_np)

    # Apply transformation
    img_tensor = flask_transform_pipeline(img).unsqueeze(0).to(device)

    # Get prediction from the model
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        top3_prob = top3_prob.cpu().numpy().flatten()
        top3_catid = top3_catid.cpu().numpy().flatten()

    top3_results = [(int(top3_catid[i]), float(top3_prob[i])) for i in range(3)]
    return render_template('index.html',
                                  prediction=top3_results[0][0],  # top-1
                                  val_acc=STATIC_VAL_ACC,
                                  top3=top3_results)

@app.route("/predict_live", methods=["POST"])
def predict_live():
    image_data = request.form.get("image_data")
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Remove the "data:image/png;base64," prefix
        base64_data = image_data.split(',')[1]
        img = Image.open(io.BytesIO(base64.b64decode(base64_data))).convert("L")
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 400

    # Optional: invert colors if needed (MNIST digits are white on black)
    img_np = np.array(img)
    if img_np.mean() > 128:
        img = Image.fromarray(255 - img_np)

    # Apply transformation
    img_tensor = flask_transform_pipeline(img).unsqueeze(0).to(device)

    # Get prediction from the model
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        top3_prob = top3_prob.cpu().numpy().flatten()
        top3_catid = top3_catid.cpu().numpy().flatten()

    top3_results = [(int(top3_catid[i]), float(top3_prob[i])) for i in range(3)]
    
    return jsonify({
        "prediction": top3_results[0][0],
        "top3": top3_results
    })


if __name__ == "__main__":
    app.run(debug=config.FLASK_DEBUG)