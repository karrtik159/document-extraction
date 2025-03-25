import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pdf2image import convert_from_path
import re
from PIL import Image
import os
import cv2
import numpy as np
from deepface import DeepFace
import logging
from utils import file_exists, read_yaml

# Load Model & Processor
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else "auto"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch_dtype, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "ekyc_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    filemode="a",
)

config_path = "config.yaml"
config = read_yaml(config_path)
artifacts = config["artifacts"]
output_path = artifacts["INTERMIDEIATE_DIR"]

# Enhance Aadhaar Image Clarity
def enhance_image(img):
    logging.info("Enhancing image clarity...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# Extract Text from Image (Optimized for Memory)
def extract_text_from_image(image):
    try:
        image = image.convert("RGB")  # Convert to correct format
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Extract text from this image."}]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():  # Avoid unnecessary memory usage
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, generated_ids
        torch.cuda.empty_cache()
        return response
    except Exception as e:
        logging.error(f"Error in text extraction: {str(e)}")
        return "Error in text extraction."

# Extract Images from PDF (Optimize Memory)
def extract_images_from_pdf(pdf_path):
    return convert_from_path(pdf_path, dpi=150)  # Reduce DPI to lower memory usage

def validate_pan_aadhaar(text):
    pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"
    aadhaar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"  # Ensuring 12 digits
    vid_pattern = r"\b\d{16}\b"  # Ensuring 16 digits with no spaces

    pan_match = re.findall(pan_pattern, text)
    aadhaar_match = [match for match in re.findall(aadhaar_pattern, text) if len(re.sub(r"\s+", "", match)) == 12]
    vid_match = re.findall(vid_pattern, text)

    if not aadhaar_match and vid_match:
        aadhaar_status = "Masked, VID Available"
    elif aadhaar_match:
        aadhaar_status = aadhaar_match
    else:
        aadhaar_status = "Not Found"

    return {
        "PAN Found": pan_match if pan_match else "Not Found",
        "Aadhaar Found": aadhaar_status
    }


# Process Uploaded File (Optimized for CUDA)
def process_file(file):
    try:
        if file.name.endswith(".pdf"):
            images = extract_images_from_pdf(file.name)
            if images:
                # Process only the first image to save memory
                img = np.array(images[0])
                del images
                torch.cuda.empty_cache()

                enhanced_image = enhance_image(img)
                extracted_text = extract_text_from_image(Image.fromarray(enhanced_image))
                validation_results = validate_pan_aadhaar(extracted_text)

                return Image.fromarray(enhanced_image), extracted_text, validation_results, None

        else:
            image = cv2.imread(file.name)
            enhanced_image = enhance_image(image)
            extracted_text = extract_text_from_image(Image.fromarray(enhanced_image))
            validation_results = validate_pan_aadhaar(extracted_text)

            return Image.fromarray(enhanced_image), extracted_text, validation_results, None

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return None, "Error processing file.", {}, None

# Gradio Interface
demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload PAN/Aadhaar (PDF or Image)"),
    outputs=[
        gr.Image(label="Enhanced Image"),
        gr.Textbox(label="Extracted Text"),
        gr.JSON(label="PAN/Aadhaar Validation"),
        gr.Textbox(label="Extracted Face Path")
    ],
    title="Enhanced KYC Verification System",
    description="Upload a PAN/Aadhaar document, extract text using QWEN 2.5 VL, validate PAN/Aadhaar details, and detect faces.",
)

demo.launch(share=True, debug=True)
