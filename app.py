import json

import torch
import streamlit as st

from PIL import Image

from src.model import load_model
from src.transforms import get_valid_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# --------------------------------------------------
# Load classes
# --------------------------------------------------

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_trained_model():

    model = load_model(
        model_path="model/best_model.pth",
        num_classes=len(class_names),
        device=device
    )

    return model


model = load_trained_model()


# --------------------------------------------------
# Transform
# --------------------------------------------------

transform = get_valid_transform()


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🌿 Tomato Crop Disease Detection")

st.write(
    "Upload a tomato leaf image and the model "
    "will predict the most likely disease."
)


uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    if st.button("🔍 Predict Disease"):

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)


        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            top_probs, top_indices = torch.topk(
                probabilities,
                k=3,
                dim=1
            )


        predicted_index = top_indices[0][0].item()

        predicted_class = class_names[
            predicted_index
        ]

        confidence = top_probs[0][0].item()


        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        st.subheader(
            "Top 3 Predictions"
        )


        for i in range(3):

            index = top_indices[0][i].item()

            probability = top_probs[0][i].item()

            st.write(
                f"{i + 1}. "
                f"{class_names[index]} — "
                f"{probability * 100:.2f}%"
            )

st.info(
    "This model provides an AI-based image classification result "
    "and should be used as a decision-support tool, not as a "
    "definitive agricultural diagnosis."
)