import os
import tempfile
import torch
import streamlit as st
from PIL import Image

from model.generation import image_caption
from model.trainer import vocab
# from model.model_config import model


# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"



# -----------------------------
# Load Model (Runs Once)
# -----------------------------
@st.cache_resource
def load_model():
    # this will load first the archticher and it's configuration
    from model.model_config import model

    # then loading the checkpoint
    checkpoint_path = "checkpoints/checkpoint_9.pth"

    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found:\n{checkpoint_path}")
        st.stop()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


model = load_model()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Captioning")
st.write("Upload an image and let the model generate a caption.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                image.save(tmp.name)

                caption = image_caption(
                    model=model,
                    full_image_path=tmp.name,
                    vocab=vocab.W2i,
                    max_length=40,
                    show_image=False
                )

            os.remove(tmp.name)

        st.success("Done!")

        st.subheader("Caption")
        st.write(caption)