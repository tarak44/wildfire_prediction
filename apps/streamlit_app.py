import streamlit as st
from PIL import Image

from wildfire_mlops.core import get_settings
from wildfire_mlops.inference import predict_image
from wildfire_mlops.modeling import load_checkpoint

settings = get_settings()

st.set_page_config(page_title="Wildfire Classifier", page_icon=None)

st.title("Wildfire Image Classifier")
st.write("Upload an image to classify wildfire vs no-wildfire.")

arch = st.selectbox(
    "Model architecture",
    options=["custom_cnn", "resnet18"],
    index=0 if settings.model_arch == "custom_cnn" else 1,
)

model_path = settings.resolve_model_path(arch)
model, class_names = load_checkpoint(
    model_path, model_arch=arch, pretrained=settings.pretrained
)

uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Could not open image. Please upload a valid image file.")
        st.text(str(e))
        st.stop()

    st.image(img, caption="Uploaded image", use_container_width=True)

    pred = predict_image(
        image=img,
        model=model,
        class_names=class_names,
        device=settings.device,
        image_size=settings.image_size,
        reference_stats_path=settings.reference_stats_path,
    )

    st.markdown("### Prediction")
    st.write(f"Class: `{pred.class_name}`")
    st.write(f"Confidence: {pred.confidence:.4f}")
    st.markdown("Class probabilities:")
    st.json(pred.probabilities)
