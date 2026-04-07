import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import h5py
import io
import os
import gdown
import time

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wheat Disease Classifier",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1QaTCsBV1gbo3kmpBU8aRf4TabX3EES_N"
MODEL_PATH     = "ConvNeXt_model.h5"
NUM_CLASSES    = 15
IMG_SIZE       = 224

FALLBACK_CLASSES = [
    "Healthy",
    "Septoria Leaf Blotch",
    "Stripe Rust (Yellow Rust)",
    "Leaf Rust (Brown Rust)",
    "Powdery Mildew",
    "Fusarium Head Blight",
    "Tan Spot",
    "Loose Smut",
    "Common Bunt",
    "Barley Yellow Dwarf",
    "Take-All Root Rot",
    "Crown Rot",
    "Sharp Eyespot",
    "Black Point",
    "Ergot",
]

# ── Model definition (must match training) ─────────────────────────────────
class ConvNeXt(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = models.convnext_large(weights=None)
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# ── Image transform ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Download model from Google Drive ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Google Drive (first run only)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    with st.spinner("🔧 Loading model weights..."):
        with h5py.File(MODEL_PATH, "r") as f:
            raw         = f["state_dict"][:]
            buf         = io.BytesIO(raw.tobytes())
            state_dict  = torch.load(buf, map_location=device)
            num_classes = int(f.attrs.get("num_classes", NUM_CLASSES))
            try:
                class_names = list(f.attrs["class_names"])
            except Exception:
                class_names = FALLBACK_CLASSES[:num_classes]

        model = ConvNeXt(num_classes=num_classes).to(device)
        model.load_state_dict(state_dict)
        model.eval()

    return model, class_names, device

# ── Inference ──────────────────────────────────────────────────────────────
def predict(image: Image.Image, model, class_names, device):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top5_idx  = probs.argsort()[::-1][:5]
    top5      = [(class_names[i], float(probs[i])) for i in top5_idx]
    return top5

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Wheat-haim.jpg/320px-Wheat-haim.jpg",
             use_column_width=True)
    st.markdown("## 🌾 About")
    st.markdown(
        "This app uses a **ConvNeXt-Large** model fine-tuned on wheat plant images "
        "to classify **15 wheat diseases**.\n\n"
        "Upload a leaf / plant image and get an instant prediction."
    )
    st.markdown("---")
    st.markdown("**Model:** ConvNeXt-Large")
    st.markdown("**Classes:** 15")
    st.markdown("**Test accuracy:** 91.9%")
    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io) · [PyTorch](https://pytorch.org)")

# ── Main ───────────────────────────────────────────────────────────────────
st.title("🌾 Wheat Disease Classifier")
st.markdown("Upload a wheat plant image to identify the disease.")

# Load model once
model, class_names, device = load_model()
st.success(f"✅ Model ready — running on **{str(device).upper()}**")

# Upload
uploaded = st.file_uploader(
    "Choose an image (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Classifying..."):
            start   = time.time()
            results = predict(image, model, class_names, device)
            elapsed = time.time() - start

        top_label, top_prob = results[0]

        st.markdown(f"### 🏆 Prediction")
        st.markdown(f"**{top_label}**")
        st.progress(top_prob)
        st.markdown(f"`{top_prob*100:.1f}%` confidence &nbsp;·&nbsp; `{elapsed*1000:.0f} ms`")

        st.markdown("---")
        st.markdown("#### Top 5 predictions")
        for label, prob in results:
            bar_color = "🟩" if prob == top_prob else "🟦"
            st.markdown(f"{bar_color} **{label}** — {prob*100:.1f}%")
            st.progress(float(prob))

    # Confidence table
    with st.expander("📊 Full probability table"):
        import pandas as pd
        all_probs = predict(image, model, class_names, device)
        df = pd.DataFrame(all_probs, columns=["Disease", "Confidence"])
        df["Confidence"] = df["Confidence"].map(lambda x: f"{x*100:.2f}%")
        st.dataframe(df, use_container_width=True)
else:
    st.info("👆 Upload an image to get started.")
