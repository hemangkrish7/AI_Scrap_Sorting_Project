# -*- coding: utf-8 -*-


import onnxruntime as ort
import os
import numpy as np
import cv2  # This is the OpenCV library
import pandas as pd
import time

# --- SETUP ---
# Define paths
MODEL_PATH = '../models/scrap_classifier_v1.onnx'
IMAGE_DIR = '../data/test_images/' # Make sure to create this folder and add some test images!
RESULTS_PATH = '../results/results.csv'
CONFIDENCE_THRESHOLD = 0.75 # Set a threshold for low confidence warnings

# The class names must be in the same order as the training data
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load the ONNX model and create an inference session
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- HELPER FUNCTIONS ---
def preprocess_image(image_path):
    """Loads, resizes, and normalizes an image for the model."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    # Add a batch dimension -> (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img

def softmax(x):
    """Compute softmax values for a set of scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

# --- MAIN SIMULATION LOOP ---
print("--- Starting Scrap Classifier Simulation ---")

results = []
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)

    # 1. Preprocess the image
    input_tensor = preprocess_image(image_path)

    # 2. Run inference
    outputs = session.run(None, {input_name: input_tensor})
    scores = outputs[0]

    # 3. Get prediction and confidence
    probabilities = softmax(scores)
    predicted_class_id = np.argmax(probabilities)
    confidence = np.max(probabilities)
    predicted_class_name = CLASS_NAMES[predicted_class_id]

    # 4. Log to console
    print(f"Image: {image_file} -> Predicted: {predicted_class_name} | Confidence: {confidence:.4f}")

    # 5. Check for low confidence
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"  -> WARNING: Low confidence prediction!")

    # 6. Store result
    results.append({
        'filename': image_file,
        'predicted_class': predicted_class_name,
        'confidence': confidence
    })

    time.sleep(1) # Simulate a 1-second delay between items on the conveyor

# --- SAVE RESULTS ---
df = pd.DataFrame(results)
df.to_csv(RESULTS_PATH, index=False)

print("\n--- Simulation Complete ---")
print(f"Results saved to {RESULTS_PATH}")
