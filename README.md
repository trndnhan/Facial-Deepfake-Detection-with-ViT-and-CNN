----------
# Facial-Deepfake-Detection-with-ViT-and-ResNet50 🎭🔍
_This repository is part of my thesis project on deepfake detection using Vision Transformers (ViT) and Convolutional Neural Networks (CNN), both with and without Low-Rank Adaptation (LoRA)._

_💜 This project has not finished yet, make sure to check this repo once in a while for further updates. I will update the training progress in the notebook file everyday and the evaluation statistic and graph after each epoch._

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

_The models are not the final version, further training iterations may help refine the performance of the models._

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

 --------
 
## 📊 Current Insight on Performance (5th epoch)

### 🎓 During training:
![image](https://github.com/user-attachments/assets/ce0db142-2a5c-4b89-addb-f8c2f448ce7f)
![image](https://github.com/user-attachments/assets/ae59e4e2-0f65-4a49-9022-be0bbff1fedb)
![image](https://github.com/user-attachments/assets/af357d06-8ed4-4e03-8bed-977b8a4ecd18)

-   **vit_lora** is the most accurate model, achieving the highest overall **Accuracy (0.9559)**, **F1 Score (0.9561)** and lowest **Loss (0.276)**. But this model, at the same time, took the logest time to train (52.42 hours) and was the most computationally expensive.
- **rn_base (Accuracy: 0.7048, Loss: 0.6001, F1 Score: 0.698)** performs the worst across all metrics, despite having the shortest train time **(6.19 hours)** and being least computationally expensive.
- Both **rn_lora (Accuracy: 0.8014, Loss: 0.4916, F1 Score: 0.8018)** and **vit_base (Accuracy: 0.8243, Loss: 0.4653, F1 Score: 0.8259)** are relatively stable but do not perform as well as **vit_lora** and as bad as **rn_base**.
- Both **LoRA-injected models** are performing better than their base respective models. **ViT**, in general, still beats **ResNet** both **with and without LoRA** in terms of performance metrics, but is significantly more computational expensive and has much longer train time.


### 🧑‍💻 During evaluation:
![image](https://github.com/user-attachments/assets/06c13878-c12d-476e-80b0-cc1e3c63e049)
![image](https://github.com/user-attachments/assets/c329a0d7-ef1a-4744-b9db-ab00d720cb43)

-   The metrics from validation set indicate that the ViT models are generalizing well to unseen data and not overfitting, while the ResNet models are biased toward negative class (fake).
-   **vit_lora** (Accuracy: 0.9736) maintains the same metrics while training and is the best-performance model. **rn_base** (0.7027) has the worst accuracy and shows signs of bias toward the negative class, as well as **rn_lora** (0.7574) despite having a bit higher accuracy. **vit_base** (0.8447) lies in the middle in term of accuracy.
-   Overall, the ViT models have strong consistency between training and validation, but not for the ResNet models.

*These remarks are not the final version, further training iterations may help refine the performance of the models.*

----------

## 🔗 Fine-Tuned Models on Hugging Face
You can access my current fine-tuned models (2 epochs) directly on Hugging Face:

-   [Fine-Tuned ViT Model](https://huggingface.co/1ancelot/vit_base)
-   [Fine-Tuned ResNet50 Model](https://huggingface.co/1ancelot/rn_base)
-   [Fine-Tuned LoRA-injected ViT Model](https://huggingface.co/1ancelot/vit_lora)
-   [Fine-Tuned LoRA-injected ResNet50 Model](https://huggingface.co/1ancelot/rn_lora)

Feel free to explore the models and their capabilities!

_The models are not the final version, further training iterations may help refine the performance of the models._

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
