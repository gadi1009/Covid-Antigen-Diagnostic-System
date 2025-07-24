# Automated COVID-19 Antigen Test Diagnostic System

This project presents a complete, two-stage deep learning system for the automated diagnosis of COVID-19 antigen test results from images. The system is designed to identify the test kit in a raw image and then classify its result into one of three categories: **POSITIVE**, **NEGATIVE**, or **INVALID**.

The project was developed as part of a deep learning course and demonstrates an end-to-end pipeline, from data collection and cleaning to the training, evaluation, and analysis of state-of-the-art models.

---

## System Architecture

The system is built as a two-stage pipeline:

1.  **Stage 1: Object Detection (YOLO-based)**
    * A YOLOv11n model, trained on a custom dataset of 966 images, is used to detect and crop the antigen test device from the input image.
    * This model achieved a **mAP50-95 of 0.905**, demonstrating high accuracy in localizing the test kit.
    * *See the `/stage1_object_detection` directory for full details.*

2.  **Stage 2: Classification (Swin Transformer V2)**
    * The cropped image from Stage 1 is passed to a state-of-the-art Swin Transformer V2 model.
    * This model was fine-tuned on a curated dataset of cropped images, including a dedicated `INVALID` class.
    * The final classifier achieved an outstanding **overall accuracy of 96.9%**, with a perfect (1.00) recall for the critical `INVALID` class.
    * *See the `/stage2_classification` directory for full details.*

---

## Final Performance

The final integrated system is highly accurate and robust. The classifier's performance on the validation set is summarized below:

| Class    | Precision | Recall | F1-Score |
| :------- | :-------- | :----- | :------- |
| INVALID  | 0.71      | **1.00** | 0.83     |
| NEGATIVE | **1.00** | 0.92   | 0.96     |
| POSITIVE | 0.99      | 0.99   | 0.99     |
|          |           |        |          |
| **Overall Accuracy:** | **96.89%** |        |          |


## Technology Stack
* Python & Jupyter Notebook
* PyTorch
* Ultralytics (YOLO)
* timm (PyTorch Image Models)
* scikit-learn
* OpenCV
* Google Colab with NVIDIA A100 GPU

---

## Repository Structure

* **/stage1_object_detection:** Contains the Jupyter Notebook and details for the YOLO object detection model.
* **/stage2_classification:** Contains the Jupyter Notebook and details for the Swin Transformer classification model.

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/2427377e-ac8b-4b44-b2bf-8fca1ff8c9b2" />
<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/6e1c7aef-acf7-43f9-b8f8-6df72d7ce1c9" />
![val_batch2_pred](https://github.com/user-attachments/assets/1a8d63d3-6447-46cf-91f2-96473b9dfc32)
