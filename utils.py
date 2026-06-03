import torch
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
from model import MultiLevelMiTB3Fusion

def load_model(weights_path, device="cpu"):
    """
    Instantiate the MultiLevelMiTB3Fusion model and load the trained weights.
    """
    model = MultiLevelMiTB3Fusion(fusion_ch=320, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    # The weights are nested under 'model_state_dict'
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    """
    Resize to (256, 256), convert to RGB, and apply ImageNet normalization.
    Returns a PyTorch tensor with shape (1, 3, 256, 256).
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to 256x256
    img_resized = image.resize((256, 256), Image.Resampling.BILINEAR)
    
    # Convert to float array [0.0, 1.0]
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # HWC -> CHW and add batch dimension
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor

def postprocess_output(logits: torch.Tensor, original_size, threshold=0.5):
    """
    Convert logits to probability, interpolate to original size, and threshold to binary mask.
    Returns:
        prob_mask: float numpy array of probabilities [0, 1]
        binary_mask: uint8 numpy array with values 0 or 255
    """
    # Sigmoid to get probabilities [0, 1]
    probs = torch.sigmoid(logits) # shape (1, 1, H, W)
    
    # Interpolate to original size
    probs_resized = F.interpolate(probs, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)
    
    # Convert to numpy
    prob_mask = probs_resized.squeeze().cpu().numpy()
    
    # Threshold to binary
    binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
    return prob_mask, binary_mask

def generate_overlay(original_image: Image.Image, binary_mask: np.ndarray, color=(244, 63, 94), alpha=100):
    """
    Overlay a binary mask on top of the original image with transparency and a glowing border.
    color: RGB tuple for the mask overlay (default: Rose/Coral red)
    alpha: transparency value (0-255)
    """
    # Ensure original is RGB
    if original_image.mode != "RGB":
        original = original_image.convert("RGB")
    else:
        original = original_image.copy()

    # Create mask image
    mask_img = Image.fromarray(binary_mask).convert("L")
    
    # Create a solid color image for overlay
    solid_color = Image.new("RGB", original.size, color=color)
    
    # Composite the solid color onto the original image using the binary mask as alpha
    # Since we want it to be semi-transparent, we modulate the mask opacity
    mask_np = np.array(mask_img)
    alpha_mask_np = (mask_np * (alpha / 255.0)).astype(np.uint8)
    alpha_mask = Image.fromarray(alpha_mask_np)
    
    overlay = Image.composite(solid_color, original, alpha_mask)
    
    # Draw outline for better aesthetics
    try:
        from PIL import ImageFilter
        # Find edges of binary mask to make a clean outline
        edges = mask_img.filter(ImageFilter.FIND_EDGES)
        # Dilate outline slightly for thickness
        edges = edges.filter(ImageFilter.MaxFilter(3))
        # Composite outline in full color
        solid_outline = Image.new("RGB", original.size, color=color)
        overlay = Image.composite(solid_outline, overlay, edges)
    except Exception as e:
        print("Outline rendering skipped:", e)
        
    return overlay

def compute_metrics(binary_mask: np.ndarray):
    """
    Computes diagnostic metrics based on the predicted binary mask.
    """
    total_pixels = binary_mask.size
    tumor_pixels = np.sum(binary_mask > 0)
    tumor_percentage = (tumor_pixels / total_pixels) * 100.0
    
    # Detection threshold (minimum tumor size to filter out noise)
    detected = tumor_pixels > 50
    
    return {
        "detected": bool(detected),
        "tumor_percentage": float(round(tumor_percentage, 2)),
        "tumor_pixels": int(tumor_pixels)
    }
