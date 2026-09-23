import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms
import urllib.request
import os
MODEL_URL = "https://github.com/dhruveesalian/UCF-Crime-Abnormal-Behavior-Detection/releases/download/v1.0/resnet18_ucf_crime_baseline.pth"

MODEL_PATH = "resnet18_ucf_crime_baseline.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Loading trained model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 1)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
st.set_page_config(
    page_title="UCF-Crime Abnormal Behaviour Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 UCF-Crime Abnormal Behaviour Detection")
st.write("Upload an image to detect whether the scene is Normal or Abnormal.")

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔎 Analyze Image"):
        st.info("Model prediction will appear here.")
