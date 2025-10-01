# AI Scrap Sorting Simulation

## Overview

This project is an end-to-end machine learning pipeline that simulates the real-time classification of scrap materials from images. The goal was to build a complete system, from data preparation and model training to lightweight deployment and a final "conveyor belt" simulation, demonstrating a full model lifecycle.


---

## Dataset

This project uses the **TrashNet** dataset, created by Gary Thung and Mindy Yang. It contains thousands of images of trash, sorted into different material classes.

The dataset can be found on platforms like GitHub and Kaggle.

* **Original Source:** [TrashNet on GitHub](https://github.com/garythung/trashnet)



---

## Model Architecture

* **Base Model**: The classification model is built using a **MobileNetV2** architecture, pre-trained on the ImageNet dataset.
* **Training Process**: **Transfer learning** was employed to accelerate training and improve performance.  The core convolutional layers of MobileNetV2 were frozen, and a new classification head (a `GlobalAveragePooling2D` layer and a `Dense` layer) was added and trained on the TrashNet data. This process retrains only the final layers of the model to specialize in recognizing our specific scrap classes.

---

## Deployment Decisions

For deployment, the model was converted from its original TensorFlow Keras format to the **ONNX (Open Neural Network Exchange)** format.  This was done to create a lightweight, highly optimized, and portable model suitable for fast inference in a real-time application. The ONNX model can be run efficiently on various platforms using the ONNX Runtime.

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
    The script will classify each image, print the results to the console, and save a `results.csv` file in the `/results` folder. 

---

## Folder Structure

The project is organized into the following directories as required:
* `/src`: Contains all source code, including the final `simulate.py` script. 
* `/data`: Holds the dataset and test images. 
* `/models`: Stores the trained Keras model and the converted `.onnx` version. 
* `/results`: Contains the output `results.csv` file from the simulation. 
