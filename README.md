# OncoVision: Deep Learning Breast Cancer Segmentation & Diagnostics

OncoVision is an interactive medical imaging web application powered by a deep learning semantic segmentation model. The application detects and highlights anomalous tumor boundaries in breast ultrasound scans in real-time, providing diagnostic metrics for clinical review.

The backend uses a custom **Multi-Level MiT-B3 Fusion** neural network architecture in PyTorch, loaded with pre-trained weights from extensive training on multiple clinical breast ultrasound datasets.

---

## 🔬 Model Architecture & Training

The core neural network (`MultiLevelMiTB3Fusion`) is designed to capture fine-grained tumor boundaries by integrating multi-scale representations:
1. **MixVisionTransformer (MiT-B3) Encoder**: Pre-trained on ImageNet to capture robust global and local context.
2. **Stage Enhancers**: Feature maps from the deepest 4 stages are passed through dilated Depthwise Separable Convolution (DSC) bundles and Channel/Spatial Attention Blocks (CAB/SAB) to enhance signal resolution defensively.
3. **Top-Down Fusion**: Multi-level features are unified and fused top-down to refine target segmentations.
4. **Bilinear Interpolation**: Bilinear resizing maps logits directly to the input dimensions.

### Datasets Trained On
The model has been optimized on three prominent open-source breast ultrasound datasets:
- **BUSI (Combined)**: [Dataset Link](https://www.kaggle.com/datasets/srivarshachivukula/busi-combined)
- **BUS-BRA**: [Dataset Link](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset)
- **Breast Dataset**: [Dataset Link](https://www.kaggle.com/datasets/hudson777/breast-dataset)

---

## ⚡ Key Features

- **Ingestion & Diagnostics**: Drag and drop breast ultrasound scans (PNG, JPG, JPEG) to run segmentation.
- **Interactive Visualizer**: Toggle between **Overlay view**, side-by-side **Comparison view**, or the raw **Binary Mask**.
- **Real-Time Aesthetic Controls**: Adjust overlay opacity and choose custom overlay colors (e.g., Neon Rose, Cyan, Emerald Green) with direct UI preview.
- **Metrics Dashboard**: Computes diagnostic statistics including diagnosis detection status (TUMOR DETECTED vs NO TUMOR DETECTED), tumor surface area ratio (%), and exact pixel coverage.
- **Local Sandbox Execution**: Completely self-contained backend running on Flask and PyTorch (runs on CPU or CUDA-capable GPUs).
- **Download Deliverables**: Export predicted tumor overlays or binary masks directly from the browser.
- **Built-in Demo Samples**: Clickable normal and pathological sample scans for instant testing.

---

## 🛠️ Installation & Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. A GPU with CUDA support is recommended but not required.

### 2. Clone and Install Dependencies
Navigate to the project directory and install the required libraries:
```bash
# Optional: Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Weights Setup
Since the trained weights file is ~586MB (too large for standard GitHub file limits), it is excluded from Git tracking via `.gitignore`. 
- Ensure you have the weights file: `MiT_fusion_stageEnhancer_BUSI (1).pth`
- Place this file directly in the **root** of the project directory.

### 4. Run the Web Application
Start the Flask local development server:
```bash
python app.py
```
Open your browser and navigate to:
[http://localhost:5000](http://localhost:5000)

---

## 🚀 Pushing to GitHub

To upload this project to your own GitHub repository, follow these steps:

### 1. Initialize Git Locally
If git is not already initialized, run the following commands in the project directory:
```bash
# Initialize local repo
git init

# Add files to staging (excluding the large .pth file ignored in .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: OncoVision Web UI and Model Architecture"
```

### 2. Create the GitHub Repository
1. Go to [github.com](https://github.com/) and log in.
2. Click the **New** button to create a new repository.
3. Set the Repository Name (e.g., `oncovision-breast-cancer-ui`).
4. Keep the repository public or private, and do **NOT** check "Add a README", "Add .gitignore", or "Choose a license" (since we have already created these).
5. Click **Create repository**.

### 3. Link and Push
Copy the commands from the GitHub repository setup page under *"…or push an existing repository from the command line"*:
```bash
# Link the remote repository (Replace with your actual GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/oncovision-breast-cancer-ui.git

# Set default branch to main
git branch -M main

# Push files to GitHub
git push -u origin main
```
Your code and UI will now be available on your GitHub profile!
