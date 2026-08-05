import os
import tempfile
import torch
import streamlit as st
from PIL import Image

from model.model import encoder_decoder
from model.generation import image_caption
from model.trainer import flick, vocab_size


# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load Model (Runs Once)
# -----------------------------
@st.cache_resource
def load_model():

    model = encoder_decoder(
        em_size=712,
        hidden_size=712,
        expand_scale=2,
        vocab_size=vocab_size,
        num_expert=3,
        max_length=1000,
        num_head=8,
        head_dim=712,
        num_layers=4,
        dropout_size=0.2
    ).to(device)

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
                    vocab=flick.vocab.W2i,
                    max_length=40,
                    show_image=False
                )

            os.remove(tmp.name)

        st.success("Done!")

        st.subheader("Caption")
        st.write(caption)