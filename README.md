----------
# Facial-Deepfake-Detection-with-ViT-and-ResNet50 🎭🔍
_This repository is part of my thesis project on deepfake detection using Vision Transformers (ViT) and Convolutional Neural Networks (CNN), both with and without Low-Rank Adaptation (LoRA)._

----------


## 🌟 Project Overview

### 🧠 Models:

-   **Vision Transformer (ViT)**: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
-   **Convolutional Neural Network (ResNet50)**: [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
-   **Training Framework**: HuggingFace's Trainer
-   **Dataset**: 140k images from a Kaggle dataset (70k real, 70k fake). ([140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces))

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
├── uploads/                     # Sample images
│
├── templates/
│   └── index.html                 # HTML template for the web demo
│
├── Thesis_Notebook.ipynb           # Jupyter notebook containing model training code.
│
├── backend.py                      # FastAPI backend for serving the demo
│
├── requirements.txt                # List of required packages
│
├── training.csv                # Training metrics
```

----------

## 🔗 Fine-Tuned Models on Hugging Face
You can access my current fine-tuned models (2 epochs) directly on Hugging Face:

-   [Fine-Tuned ViT Model](https://huggingface.co/1ancelot/vit_base)
-   [Fine-Tuned ResNet50 Model](https://huggingface.co/1ancelot/rn_base)
-   [Fine-Tuned LoRA-injected ViT Model](https://huggingface.co/1ancelot/vit_lora)
-   [Fine-Tuned LoRA-injected ResNet50 Model](https://huggingface.co/1ancelot/rn_lora)

Feel free to explore the models and their capabilities!

----------

## 🚀 Web Demo Overview

Curious about the demo but don’t want to run it yourself? Check out the demo in action below! 👇

https://github.com/user-attachments/assets/d49c4e7d-727c-49de-8fe2-439f71296e3e

### ✨ Features:

-   Upload or generate an image using **StyleGAN3**.
-   Choose the model (ViT or ResNet50) to classify whether the image is **real** or a **deepfake**.
-   Predictions are displayed directly on the web interface.
----------

## 🔧 Installation & Usage

To get started with the repository, you can follow these steps:

1.  **Clone the repository**:
    
   ``` bash
    git clone https://github.com/your-username/Facial-Deepfake-Detection-with-ViT-and-ResNet50.git
    cd Facial-Deepfake-Detection-with-ViT-and-ResNet50` 
```
    
2.  **Install dependencies**:
    
   ``` bash
    `pip install -r requirements.txt` 
   ```
    
3.  **Run the web demo**:
    
   ```bash
    `uvicorn backend:app --reload` 
   ```
    
4.  Open your browser and navigate to `http://127.0.0.1:8000` to access the web interface.
 --------
## 📅 Future Updates

-   **Full comparative analysis** between ViT and ResNet50 models with/without LoRA.
-   **Thesis paper release** for public access.
-   **Final model versions** will be pushed to Hugging Face Hub.
-   **Model deployment** using cloud platform and following MLOps cycle
