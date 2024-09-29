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
```

 --------
 
## 📊 Current Insight on Performance (1st epoch)

### 🎓 During training:
-   **vit_lora** is the most accurate model, achieving the highest overall accuracy (0.638) and lowest loss (0.6387).
- **rn_lora** performs the worst (0.566 accuracy and 0.6874 loss), indicating that the chosen layer for LoRA fine-tuning might not benefit the ResNet architecture as much as it does for the Vision Transformer.
- Both **rn_base** (0.6025 accuracy and 0.6808 loss) and **vit_base** (0.6105 accuracy and 0.6645 loss) are relatively stable but do not perform as well as **vit_lora**.

### 🧑‍💻 During evaluation:
-   **rn_base** (0.61 accuracy) shows slightly better performance than **vit_base** (0.63 accuracy) in detecting real images, but **vit_base** is more balanced overall, especially for detecting fakes.
-   **rn_lora** (0.58 accuracy) suffers from a major imbalance, overfitting to fake images and severely underperforming on real images. This leads to reduced accuracy and poor recall for real images.
-   **vit_lora** (0.66 accuracy) performs significantly better than **rn_lora** and achieves the highest overall accuracy (~0.66). It also maintains a good balance between fake and real class detection, making it a better candidate for balanced detection tasks.

*These remarks are not the final version, further training iterations may help refine the performance of the models.*

----------

## 🔗 Fine-Tuned Models on Hugging Face
You can access my current fine-tuned models (1 epoch) directly on Hugging Face:

-   [Fine-Tuned ViT Model](https:/huggingface.co/1ancelot/vit_base)
-   [Fine-Tuned ResNet50 Model](https:/huggingface.co/1ancelot/rn_base)
-   [Fine-Tuned LoRA-injected ViT Model](https:/huggingface.co/1ancelot/vit_lora)
-   [Fine-Tuned LoRA-injected ResNet50 Model](https:/huggingface.co/1ancelot/rn_lora)

Feel free to explore the models and their capabilities!

----------

## 📜 Thesis Paper 📝
-   **Title**: _Comparative Analysis of Vision Transformers and Convolutional Neural Networks for Deepfake Detection in Human Faces with Low-Rank Adaptation_
-   The thesis paper is currently **in progress** and not yet available to the public. Stay tuned for updates!
-   **Note**: While the web demo is fully functional, the models and notebooks are still **in development** and are **not final versions**.
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
