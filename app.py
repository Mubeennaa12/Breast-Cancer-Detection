import os
import io
import base64
import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from utils import load_model, preprocess_image, postprocess_output, generate_overlay, compute_metrics

app = Flask(__name__, static_folder="static", static_url_path="")

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Flask App: Using device: {device}")

# Path to model weights
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "MiT_fusion_stageEnhancer_BUSI (1).pth")

model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Model weights file not found at '{WEIGHTS_PATH}'. Please ensure the file is named correctly and placed in the project root."
            )
        print(f"Loading weights from {WEIGHTS_PATH}...")
        model = load_model(WEIGHTS_PATH, device=device)
        print("Model loaded successfully!")
    return model

def pil_to_base64(img: Image.Image, format="PNG"):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = "".join([c*2 for c in hex_str])
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image was sent
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided in request"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        original_size = image.size # (W, H)
        
        # Load weights and model
        net = get_model()
        
        # Preprocess
        input_tensor = preprocess_image(image).to(device)
        
        # Inference
        with torch.no_grad():
            logits = net(input_tensor)
            
        # Postprocess
        prob_mask, binary_mask = postprocess_output(logits, original_size)
        
        # Compute metrics
        metrics = compute_metrics(binary_mask)
        
        # Get optional parameters
        overlay_color_hex = request.form.get("color", "#f43f5e") # Coral/rose
        overlay_alpha = int(request.form.get("opacity", "100")) # 0-255
        
        try:
            overlay_color = hex_to_rgb(overlay_color_hex)
        except Exception:
            overlay_color = (244, 63, 94)
            
        # Generate visualization images
        mask_pil = Image.fromarray(binary_mask).convert("L")
        overlay_pil = generate_overlay(image, binary_mask, color=overlay_color, alpha=overlay_alpha)
        
        # Convert to Base64 data URLs
        orig_base64 = pil_to_base64(image.convert("RGB"), format="JPEG")
        mask_base64 = pil_to_base64(mask_pil, format="PNG")
        overlay_base64 = pil_to_base64(overlay_pil, format="PNG")
        
        # Build response payload
        payload = {
            "success": True,
            "detected": metrics["detected"],
            "tumor_percentage": metrics["tumor_percentage"],
            "tumor_pixels": metrics["tumor_pixels"],
            "images": {
                "original": orig_base64,
                "mask": mask_base64,
                "overlay": overlay_base64
            }
        }
        return jsonify(payload)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    # Simple check if weights exist
    weights_exist = os.path.exists(WEIGHTS_PATH)
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "weights_loaded": model is not None,
        "weights_exist": weights_exist
    })

if __name__ == '__main__':
    # Initialize the model on startup so the first request is fast
    try:
        get_model()
    except Exception as e:
        print(f"Warning: Model could not be loaded on startup ({e}). It will load on first request.")
        
    app.run(host='0.0.0.0', port=5000, debug=True)
