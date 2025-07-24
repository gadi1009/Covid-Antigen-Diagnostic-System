# Stage 2: Classification Model (Swin Transformer V2)

This directory contains the implementation of the second and final stage of the project: a state-of-the-art model for classifying the result of a cropped COVID-19 antigen test.

## Objective
The goal was to take the cropped image from Stage 1 and classify it with high accuracy into one of three categories: `POSITIVE`, `NEGATIVE`, or `INVALID`.

## Methodology

1.  **Dataset:** The training dataset was created by processing the source images with the YOLO model from Stage 1. A critical part of this stage was **data curation**: I manually collected, cleaned, and integrated over 170 images for the `INVALID` class to create a robust and balanced 3-class dataset.

2.  **Model:** After establishing a baseline with a ResNet50 model (achieving ~71% accuracy), I upgraded the architecture to a state-of-the-art **Swin Transformer V2** model to maximize performance. The model was trained using transfer learning, leveraging weights pre-trained on ImageNet.

3.  **Training:** The model was trained using extensive **Data Augmentation** (random flips, rotations, color jitter) to prevent overfitting and improve the model's ability to generalize.

## Results

The final Swin Transformer model achieved an outstanding **overall accuracy of 96.9%**.

The model demonstrated near-perfect performance on `POSITIVE` and `NEGATIVE` cases. Most importantly, it achieved a **recall of 1.00 for the `INVALID` class**, meaning it correctly identified 100% of the faulty tests in the validation set, making it a very safe and reliable diagnostic tool.

Advanced explainability techniques like **Grad-CAM** were also implemented to visually confirm that the model was focusing on the correct regions (the C and T lines) when making its predictions.
