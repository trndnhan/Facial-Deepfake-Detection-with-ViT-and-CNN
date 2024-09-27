# 🌟 Facial Deepfake Detection with ViT and ResNet50 🌟

🚀 **Thesis Project Repository** 🚀

Welcome to the **Facial-Deepfake-Detection-with-ViT-and-ResNet50** repository, part of my AI thesis project! The primary goal of this project is to provide a **comparative analysis** between two powerful architectures: **Vision Transformer (ViT)** and **ResNet50**, both with and without **LoRA** (Low-Rank Adaptation), in the challenging task of classifying facial portraits as either **real** or **deepfaked**.

---

## 🧠 Project Overview

This project aims to tackle the **Deepfake detection problem** by training and comparing two state-of-the-art models—ViT and ResNet50. The analysis involves both base models and their LoRA-adapted versions to understand how each handles the complexities of deepfake image detection. The dataset and models are processed and managed using the **HuggingFace Trainer**, and the trained models are available on the **HuggingFace Hub**.

### Key Goals:
- **Classify** an image as either a **deepfake** or a **real** facial portrait.
- Provide **comparative results** for **ViT** vs **ResNet50**, with and without **LoRA**.
- Build a **FastAPI-based web demo** for model inference.

---

## 📂 Repository Structure
📦 Facial-Deepfake-Detection-with-ViT-and-ResNet50 
├── stylegan3/ # Pulled from NVIDIA's StyleGAN3 repository 
├── templates/ 
│ └── index.html # Webpage template for FastAPI demo 
├── Thesis_Notebook.ipynb # Full Jupyter notebook with training/validation analysis 
├── backend.py # FastAPI backend for model inference 
├── requirements.txt # Python dependencies


---

## 🧑‍💻 Dataset

The dataset used for this project is a **100k image set** from Kaggle, consisting of:
- **50k real** images
- **50k fake** images

### Data Split:
- **80% Train** (80k images)
- **10% Validation** (10k images)
- **10% Test** (10k images)

Due to computational limitations on Google Colab (CPU), only **1k real** and **1k fake** images are trained at a time. The model is trained for **0.875 epoch**, effectively covering the 70k training images.

---

## 📊 Latest Training Stats (at 0.875 Epoch)

Here are the key metrics for the four models trained up to 0.875 epochs. The detailed statistics and graphs are available in the **[Thesis Notebook](./Thesis_Notebook.ipynb)**:

- **Base ResNet50**: Accuracy: **61.10%**, F1-Score: **0.5918**
- **Base ViT**: Accuracy: **60.05%**, F1-Score: **0.5971**
- **LoRA ResNet50**: Accuracy: **56.65%**, F1-Score: **0.5328**
- **LoRA ViT**: Accuracy: **64.75%**, F1-Score: **0.6455**

### 📝 Analysis:

- **LoRA ViT** performs the best with a balanced **64.75% accuracy** and **0.6455 F1-Score**, showing good capability in detecting deepfakes.
- **Base ResNet** and **Base ViT** are relatively close, but ResNet is marginally ahead with a slightly better F1-Score.
- **LoRA ResNet** shows weaker performance, especially in terms of F1-Score, indicating that LoRA may not be as effective for ResNet in this domain.

To dive deeper into these results, check out the **detailed training logs** in the notebook! 📔

---

## 🧪 Latest Validation Stats

Here are the results for the models on the **validation set** (2k images):

- **Base ResNet50**: Accuracy: **62%** (Real detection is better than Fake)
- **Base ViT**: Accuracy: **61%** (Balanced precision between classes)
- **LoRA ResNet50**: Accuracy: **54%** (Very high recall for fakes, but poor performance on real)
- **LoRA ViT**: Accuracy: **66%** (Best performer with solid F1-Score across both classes)

### 📝 Analysis:

- **LoRA ViT** outperforms all models on validation, showing consistent results across both fake and real images.
- **Base ViT** performs well with balanced detection, while **Base ResNet** shows a significant bias toward detecting real images over fakes.
- **LoRA ResNet** struggles with detecting real images, suffering from a high false positive rate for fakes.

For full validation stats, including **confusion matrices** and **classification reports**, refer to the **notebook**.

---

## 🌐 Demo Website

A **demo website** has been developed to showcase the model’s inference capabilities. The web application is built using **FastAPI** and allows users to either **upload an image** or use a **StyleGAN3-generated image** for model inference. Users can select between the **ViT** and **ResNet50** models, both with and without LoRA.


----------

## 🛠️ Installation and Usage

To set up this repository on your local machine, follow these steps:

1.  Clone the repository:
```bash
$ git clone https://github.com/your-username/Facial-Deepfake-Detection-with-ViT-and-ResNet50.git
```
2. Install dependencies:
```bash
$ pip install -r requirements.txt
```
3. Start the FastAPI server for the demo:
```bash
$ uvicorn backend:app --reload
```
4. Then visit the web interface at **http://127.0.0.1:8000/** and upload an image or generate one using **StyleGAN3**!

---
## 🏆 Key Insights

-   **LoRA ViT** demonstrates the best performance in detecting deepfakes, both during training and validation, with the highest accuracy and F1-Score.
-   **Base ResNet** shows a reasonable performance but struggles with detecting fakes effectively.
-   **LoRA ResNet** faces challenges, indicating that LoRA might not be suitable for the ResNet architecture in this task.
-   **Training on small batches** (1k real, 1k fake at a time) limits model convergence, but the results still offer valuable insights for **deepfake detection**.
