----------
# Facial-Deepfake-Detection-with-ViT-and-ResNet50 🎭🔍
_This repository is part of my thesis project on deepfake detection using Vision Transformers (ViT) and Convolutional Neural Networks (CNN), both with and without Low-Rank Adaptation (LoRA)._

----------


## 🌟 Project Overview

### 🧠 Models:

-   **Vision Transformer (ViT)**: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
-   **Convolutional Neural Network (ResNet50)**: [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
-   **Training Framework**: HuggingFace's Trainer
-   **Dataset**: 100k images from a Kaggle dataset (50k real, 50k fake). (The original dataset have 140k images but only 100k were used: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces))

### 🚀 Goals:

-   **Comparative Analysis**: Investigating the performance of ViT and ResNet50, both with and without LoRA.
-   **Image Classification Task**: Determining whether a facial image is real or a deepfake.

### 🌐 Inference Demo:

A **web-based demo** is built using **FastAPI**. This allows users to either upload an image or generate one via **StyleGAN3** and then select the model (ViT or ResNet50) for prediction. The models are hosted on HuggingFace's hub for easy access.

---

## 📁 Repository Structure

```bash
Facial-Deepfake-Detection-with-ViT-and-ResNet50/
│
├── stylegan3/                     # Pulled from NVIDIA's GitHub
│
├── templates/
│   └── index.html                 # HTML template for the web demo
│
├── Thesis_Notebook.ipynb           # Jupyter notebook containing model training code
│
├── backend.py                      # FastAPI backend for serving the demo
│
├── requirements.txt                # List of required packages
│
└── Web demo.mp4                    # Video showcasing the demo in action` 
```
----------

## 📜 Thesis Paper 📝
-   **Title**: _Comparative Analysis of Vision Transformers and Convolutional Neural Networks for Deepfake Detection in Human Faces with Low-Rank Adaptation_
-   The thesis paper is currently **in progress** and not yet available to the public. Stay tuned for updates!
-   **Note**: While the web demo is fully functional, the models and notebooks are still **in development** and are **not final versions**.
----------

## 🚀 Web Demo Overview

Curious about the demo but don’t want to run it yourself? Check out the demo in action below! 👇
(https://raw.githubusercontent.com/1ancelot/Facial-Deepfake-Detection-with-ViT-and-CNN/main/assets/Web%20demo.mp4)

### ✨ Features:

-   Upload or generate an image using **StyleGAN3**.
-   Choose the model (ViT or ResNet50) to classify whether the image is **real** or a **deepfake**.
-   Predictions are displayed directly on the web interface.
