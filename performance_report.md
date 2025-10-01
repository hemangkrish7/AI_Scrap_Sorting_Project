# Model Performance Report

## Summary

The classification model achieved a final **validation accuracy of 75%** after 10 epochs of training. The model demonstrates a solid ability to classify common scrap materials, showing that the transfer learning approach with MobileNetV2 is effective for this task.

---

## Visualizations

The following charts summarize the training process and the model's performance on the validation set.

### Accuracy and Loss Curves

These plots show the model's accuracy and loss on both the training and validation data over 10 epochs.

*[Insert your screenshot of the accuracy/loss plot here]*

### Confusion Matrix

This matrix shows the class-by-class performance and highlights which materials the model sometimes confused.

*[Insert your screenshot of the confusion matrix here]*

---

## Key Findings & Analysis

* **Strongest Performance**: The model is highly effective at identifying the **'paper'** class, achieving both high precision and recall.
* **Areas for Improvement**: The main challenge lies in distinguishing between visually similar materials, particularly **'glass', 'metal', and 'plastic'**. The confusion matrix shows that these classes were sometimes misclassified as one another.
* **Training Behavior**: The training curves indicate slight **overfitting**, as the validation accuracy began to plateau while the training accuracy continued to increase. Future work could introduce data augmentation or fine-tuning to mitigate this and potentially improve accuracy further.
