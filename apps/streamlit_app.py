import base64
from io import BytesIO

import streamlit as st
from PIL import Image

from wildfire_mlops.core import get_settings
from wildfire_mlops.inference import EnvironmentalFeatures, predict_image
from wildfire_mlops.modeling import load_checkpoint

settings = get_settings()

st.set_page_config(page_title="Wildfire Risk Intelligence", page_icon=None, layout="wide")

st.title("Wildfire Risk Intelligence")
st.write(
    "Upload an image and optionally add weather context to estimate wildfire risk, "
    "confidence, and an explainability heatmap."
)

arch_options = [
    "custom_cnn",
    "resnet18",
    "efficientnet_b0",
    "multimodal_efficientnet_b0",
]

arch = st.selectbox(
    "Model architecture",
    options=arch_options,
    index=arch_options.index(settings.model_arch) if settings.model_arch in arch_options else 0,
)

model_path = settings.resolve_model_path(arch)
model, class_names = load_checkpoint(model_path, model_arch=arch, pretrained=settings.pretrained)

with st.sidebar:
    st.subheader("Optional Context Signals")
    temperature_c = st.number_input("Temperature (C)", min_value=-20.0, max_value=60.0, value=32.0)
    humidity_pct = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=24.0)
    wind_speed_kph = st.number_input(
        "Wind speed (km/h)", min_value=0.0, max_value=150.0, value=18.0
    )
    drought_index = st.number_input("Drought index", min_value=0.0, max_value=800.0, value=320.0)
    vegetation_dryness = st.slider("Vegetation dryness", min_value=0.0, max_value=1.0, value=0.65)
    days_since_rain = st.number_input("Days since rain", min_value=0.0, max_value=90.0, value=8.0)
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.0)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-121.0)

uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Could not open image. Please upload a valid image file.")
        st.text(str(e))
        st.stop()

    environmental_features = EnvironmentalFeatures(
        latitude=latitude,
        longitude=longitude,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        wind_speed_kph=wind_speed_kph,
        drought_index=drought_index,
        vegetation_dryness=vegetation_dryness,
        days_since_rain=days_since_rain,
    )

    pred = predict_image(
        image=img,
        model=model,
        class_names=class_names,
        device=settings.device,
        image_size=settings.image_size,
        reference_stats_path=settings.reference_stats_path,
        environmental_features=environmental_features,
        include_explainability=True,
    )

    image_col, explain_col = st.columns(2)
    with image_col:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with explain_col:
        if pred.explainability and pred.explainability.overlay_base64:
            overlay = Image.open(
                BytesIO(base64.b64decode(pred.explainability.overlay_base64))
            ).convert("RGB")
            st.image(overlay, caption="Grad-CAM highlight", use_container_width=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted class", pred.class_name)
    metric_col2.metric("Image confidence", f"{pred.confidence:.2%}")
    metric_col3.metric("Overall wildfire risk", f"{pred.overall_risk_score:.2%}")

    st.markdown("### Risk summary")
    st.write(f"Risk level: `{pred.risk_level}`")
    st.write(pred.recommended_action)
    if pred.explainability and pred.explainability.summary:
        st.caption(pred.explainability.summary)

    st.markdown("### Probability breakdown")
    st.json(pred.probabilities)

    if pred.context_risk_score is not None:
        st.markdown("### Environmental context")
        st.write(f"Contextual risk score: `{pred.context_risk_score:.2%}`")
        for contributor in pred.top_contributors:
            st.write(f"- `{contributor.factor}`: {contributor.rationale}")
