import sys
import os

# Add the stylegan3 directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'stylegan3'))

# Now import the necessary modules
import dnnlib
import legacy

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
import shutil
import time
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io
from peft import PeftConfig, PeftModel

app = FastAPI()

# Set up the path to the templates directory for rendering HTML
templates = Jinja2Templates(directory="templates")

# Mount the static files directory to serve CSS
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Initialize processors
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
rn_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# Load the base models for classification
vit_base = AutoModelForImageClassification.from_pretrained("1ancelot/vit_base")
rn_base = AutoModelForImageClassification.from_pretrained("1ancelot/rn_base")

# Load the Lora models
vit_model = AutoModelForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,
    ignore_mismatched_sizes=True
)
rn_model = AutoModelForImageClassification.from_pretrained(
    'microsoft/resnet-50',
    num_labels=2,
    ignore_mismatched_sizes=True
)

vit_lora = PeftModel.from_pretrained(vit_model, "1ancelot/vit_lora")
rn_lora = PeftModel.from_pretrained(rn_model, "1ancelot/rn_lora")

# Define transformations
vit_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: vit_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0))
])

rn_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: rn_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0))
])

# Directory to save uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load StyleGAN3 pre-trained model (using FFHQ for human portraits)
STYLEGAN3_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the StyleGAN3 model from the URL
with dnnlib.util.open_url(STYLEGAN3_URL) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

# Define the maximum age of files (in seconds)
MAX_FILE_AGE = 60 * 60  # 1 hour

def cleanup_old_files():
    current_time = time.time()
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > MAX_FILE_AGE:
                os.remove(file_path)

# Helper function to generate a random latent vector (seed to vector)
def seed2vec(G, seed):
    return np.random.RandomState(seed).randn(1, G.z_dim)

# Helper function to generate an image using the StyleGAN3 model
def generate_stylegan_image(seed):
    z = seed2vec(G, seed)
    z = torch.from_numpy(z).to(device)
    label = torch.zeros([1, G.c_dim], device=device)  # Assume unconditional model
    img = G(z, label, truncation_psi=1.0, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy(), 'RGB')

# Prediction helper function
def predict(image_path, classifier_model):
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Select the model and preprocessing based on classifier_mode
    if classifier_model == "vit_base":
        model = vit_base
        processed_image = vit_transforms(image).unsqueeze(0)  # Preprocess for ViT and add batch dimension
    elif classifier_model == "rn_base":
        model = rn_base
        processed_image = rn_transforms(image).unsqueeze(0)  # Preprocess for ResNet and add batch dimension
    elif classifier_model == "vit_lora":
        model = vit_lora
        processed_image = vit_transforms(image).unsqueeze(0)
    elif classifier_model == "rn_lora":
        model = rn_lora
        processed_image = rn_transforms(image).unsqueeze(0)
    else:
        return "Invalid model selected"
    
    # Perform inference
    with torch.no_grad():
        outputs = model(processed_image)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
    
    # Map the predicted class to 'fake' or 'real'
    class_names = {0: 'fake', 1: 'real'}
    predicted_label = class_names.get(predicted_class, "Unknown")
    
    return f"Predicted Class: {predicted_label}"

# Route to classify an image
@app.post("/classify")
async def classify_image(processed_image_url: str = Form(...), classifier_model: str = Form(...)):
    # Convert the URL path to the local file system path
    local_image_path = os.path.join(UPLOAD_DIR, os.path.basename(processed_image_url))

    # Check if the file exists locally
    if not os.path.exists(local_image_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    
    # Get the prediction using the selected model
    prediction = predict(local_image_path, classifier_model)
    
    return JSONResponse(content={"prediction": prediction})

# Serve the HTML form from the frontend file
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    # Clean up old files when the server starts
    cleanup_old_files()

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, image: UploadFile = File(...)):
    image_filename = f"uploaded_{image.filename}"
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    
    # Save the uploaded file
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Open the image file
    image_file = Image.open(image_path).convert("RGB")

    # Apply transformations for ViT and ResNet
    vit_processed_tensor = vit_transforms(image_file)
    rn_processed_tensor = rn_transforms(image_file)

    # Convert tensors to PIL images for display
    vit_processed_image = transforms.ToPILImage()(vit_processed_tensor)
    rn_processed_image = transforms.ToPILImage()(rn_processed_tensor)

    # Save the processed images
    vit_processed_filename = f"vit_processed_{image_filename}"
    rn_processed_filename = f"rn_processed_{image_filename}"
    
    vit_processed_path = os.path.join(UPLOAD_DIR, vit_processed_filename)
    rn_processed_path = os.path.join(UPLOAD_DIR, rn_processed_filename)
    
    vit_processed_image.save(vit_processed_path)
    rn_processed_image.save(rn_processed_path)

    # Generate URLs for the uploaded and processed images
    uploaded_image_url = f"/uploads/{image_filename}"
    vit_processed_url = f"/uploads/{vit_processed_filename}"
    rn_processed_url = f"/uploads/{rn_processed_filename}"

    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "uploaded_image_url": uploaded_image_url,
            "vit_processed_url": vit_processed_url,
            "rn_processed_url": rn_processed_url
        }
    )

@app.get("/generate", response_class=HTMLResponse)
async def generate_image(request: Request):
    seed = np.random.randint(0, 100000)  # Generate a random seed
    generated_image = generate_stylegan_image(seed)

    # Save the generated image
    generated_image_filename = f"generated_image_{seed}.png"
    generated_image_path = os.path.join(UPLOAD_DIR, generated_image_filename)
    generated_image.save(generated_image_path)

    # Apply preprocessing using ViT and ResNet
    vit_processed_tensor = vit_transforms(generated_image)
    rn_processed_tensor = rn_transforms(generated_image)
    
    vit_processed_image = transforms.ToPILImage()(vit_processed_tensor)
    rn_processed_image = transforms.ToPILImage()(rn_processed_tensor)

    # Save the processed images
    vit_processed_filename = f"vit_processed_generated_image_{seed}.png"
    rn_processed_filename = f"rn_processed_generated_image_{seed}.png"

    vit_processed_path = os.path.join(UPLOAD_DIR, vit_processed_filename)
    rn_processed_path = os.path.join(UPLOAD_DIR, rn_processed_filename)

    vit_processed_image.save(vit_processed_path)
    rn_processed_image.save(rn_processed_path)

    # Generate URLs for the generated and processed images
    generated_image_url = f"/uploads/{generated_image_filename}"
    vit_processed_url = f"/uploads/{vit_processed_filename}"
    rn_processed_url = f"/uploads/{rn_processed_filename}"

    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "generated_image_url": generated_image_url,
            "vit_processed_url": vit_processed_url,
            "rn_processed_url": rn_processed_url
        }
    )

# Serve the uploaded images from the uploads directory
@app.get("/uploads/{filename}")
async def serve_image(filename: str):
    image_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return HTMLResponse(content="File not found", status_code=404)

# Run the FastAPI app: `uvicorn backend:app --reload`
