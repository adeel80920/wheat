# 🌾 Wheat Disease Classifier

A Streamlit web app that classifies **15 wheat diseases** using a fine-tuned **ConvNeXt-Large** model.

**Live demo:** *(add your Streamlit Cloud URL here after deployment)*

---

## Features

- Upload any wheat plant image (JPG / PNG)
- Instant prediction with confidence score
- Top-5 results with probability bars
- Full probability table for all 15 classes
- Model auto-downloads from Google Drive on first run

---

## Local setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/wheat-disease-classifier.git
cd wheat-disease-classifier

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The model (~800 MB) downloads automatically from Google Drive on the first run.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy** — done!

> **Note:** Streamlit Cloud has ~1 GB RAM on the free tier. The ConvNeXt-Large model
> is large; if you hit memory limits consider using `convnext_tiny` or `convnext_small`
> in your training notebook and re-exporting.

---

## Project structure

```
wheat-disease-classifier/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Model details

| Property | Value |
|---|---|
| Architecture | ConvNeXt-Large |
| Input size | 224 × 224 |
| Classes | 15 |
| Test accuracy | 91.9% |
| Test F1 (weighted) | 0.901 |
| Framework | PyTorch |

### Classes
1. Healthy  
2. Septoria Leaf Blotch  
3. Stripe Rust (Yellow Rust)  
4. Leaf Rust (Brown Rust)  
5. Powdery Mildew  
6. Fusarium Head Blight  
7. Tan Spot  
8. Loose Smut  
9. Common Bunt  
10. Barley Yellow Dwarf  
11. Take-All Root Rot  
12. Crown Rot  
13. Sharp Eyespot  
14. Black Point  
15. Ergot  

---

## License

MIT
