import streamlit as st
import tempfile
from pathlib import Path
from PIL import Image
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.prediction import Prediction


def load_image(uploaded_file) -> Image.Image:
    """Load an uploaded image and ensure it is in RGB format."""
    image = Image.open(uploaded_file)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    return image


def save_temp_image(image: Image.Image) -> Path:
    """Save the given image to a temporary JPEG file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")
        return Path(tmp.name)


def run_prediction(image_path: Path) -> str:
    """Run model prediction on the given image and return label."""
    config = ConfigurationManager()
    prediction_config = config.get_prediction_config()
    predictor = Prediction(config=prediction_config, img=str(image_path))
    return predictor.predict()


# -------------------- UI --------------------
st.set_page_config(page_title="Image Classification Demo", layout="wide")

# Sidebar for upload
st.sidebar.header("📂 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

st.title("🖼️ Image Classification")
st.markdown(
    "<p style='color:gray;'>Upload an image to see the model's prediction.</p>",
    unsafe_allow_html=True
)

if uploaded_file:
    image = load_image(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        temp_path = save_temp_image(image)

        if st.button("🔍 Predict", type="primary"):
            with st.spinner("Predicting... Please wait."):
                label = run_prediction(temp_path)
                
            st.markdown(
                f"""
                <div style='padding:20px; border-radius:10px; background-color:#f6f6f6; text-align:center;'>
                    <h3 style='color:#4CAF50;'>✅ Prediction</h3>
                    <p style='font-size:20px; font-weight:bold; color:#333;'>{label}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Please upload an image from the sidebar to get started.")
