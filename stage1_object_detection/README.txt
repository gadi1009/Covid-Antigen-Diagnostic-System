# Stage 1: Object Detection Model (YOLOv11n)

This directory contains the implementation of the first stage of the project: an object detection model trained to find and crop COVID-19 antigen tests from images.

## Objective
The primary goal of this stage was to build a robust model capable of accurately localizing the test kit, which serves as the input for the second-stage classification model.

## Methodology

1.  **Dataset:** I collected and curated a dataset of **966 images** from various sources. A portion was sourced from public repositories on Roboflow, but to ensure maximum quality, the majority of the dataset was manually reviewed and labeled. The data was split into a training set (725 images) and a validation set (241 images) following a 75/25 ratio.

2.  **Labeling:** Using an annotation tool, I manually created bounding boxes for each test kit. Crucially, I defined two distinct classes during this process: `class 0` for negative tests and `class 1` for positive tests. This allowed the YOLO model to learn not only the location but also a preliminary classification.

3.  **Model:** I initially explored the standard YOLOv5 architecture. However, to achieve the highest possible accuracy and utilize the latest advancements, I chose to train a **YOLOv11n** model. This model was fine-tuned for 50 epochs on the custom dataset, using automated data augmentation techniques (blur, color adjustments) to improve generalization.

## Results

The model training was highly successful, achieving an overall **mAP50-95 of 0.905** on the validation set. The final trained model, `best.pt`, is used as a critical component in the second stage of the project