import streamlit as st
from transformers import pipeline
from PIL import Image

# Set page title
st.set_page_config(page_title="AI Image Caption Generator", page_icon="🧠", layout="centered")

# Header
st.title("🧠 AI Image Caption Generator")
st.write("Upload an image and let AI describe it for you!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

pipe = load_model()

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("Generating caption... 🤖"):
        result = pipe(image)
        caption = result[0].get("generated_text") or result[0].get("caption") or "No caption generated."

    # Show result
    st.subheader("📝 Generated Caption:")
    st.success(caption)

    # Optional: Button to copy caption
    st.code(caption, language="text")
