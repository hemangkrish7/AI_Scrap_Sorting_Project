# AI Scrap Sorting Simulation

## Overview

This project is an end-to-end machine learning pipeline that simulates the real-time classification of scrap materials from images. The goal was to build a complete system, from data preparation and model training to lightweight deployment and a final "conveyor belt" simulation, demonstrating a full model lifecycle.

---

## Dataset

* [cite_start]**Dataset Used**: The project utilizes the **TrashNet** dataset. [cite: 1, 31]
* **Reasoning**: This dataset was chosen because it contains thousands of images across 6 relevant classes of materials (cardboard, glass, metal, paper, plastic, trash), which perfectly aligns with the project's scrap-sorting theme. [cite_start]The data is pre-sorted into folders by class, making it ideal for image classification tasks. [cite: 1, 31]

---

## Model Architecture

* **Base Model**: The classification model is built using a **MobileNetV2** architecture, pre-trained on the ImageNet dataset.
* [cite_start]**Training Process**: **Transfer learning** was employed to accelerate training and improve performance. [cite: 6] The core convolutional layers of MobileNetV2 were frozen, and a new classification head (a `GlobalAveragePooling2D` layer and a `Dense` layer) was added and trained on the TrashNet data. This process retrains only the final layers of the model to specialize in recognizing our specific scrap classes.

---

## Deployment Decisions

[cite_start]For deployment, the model was converted from its original TensorFlow Keras format to the **ONNX (Open Neural Network Exchange)** format. [cite: 6] This was done to create a lightweight, highly optimized, and portable model suitable for fast inference in a real-time application. The ONNX model can be run efficiently on various platforms using the ONNX Runtime.

---

## How to Run the Simulation

1.  **Clone the repository.**
2.  **Install dependencies.**
    ```bash
    pip install tensorflow onnxruntime opencv-python pandas scikit-learn seaborn
    ```
3.  **Add test images** to the `/data/test_images/` directory.
4.  **Run the script.** Navigate to the `src` directory and execute the simulation script.
    ```bash
    cd src
    python simulate.py
    ```
    [cite_start]The script will classify each image, print the results to the console, and save a `results.csv` file in the `/results` folder. [cite: 6]

---

## Folder Structure

The project is organized into the following directories as required:
* [cite_start]`/src`: Contains all source code, including the final `simulate.py` script. [cite: 12]
* [cite_start]`/data`: Holds the dataset and test images. [cite: 14]
* [cite_start]`/models`: Stores the trained Keras model and the converted `.onnx` version. [cite: 17]
* [cite_start]`/results`: Contains the output `results.csv` file from the simulation. [cite: 16]
