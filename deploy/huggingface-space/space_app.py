import base64
import json
import os
from io import BytesIO

import httpx
import streamlit as st
from PIL import Image

DEFAULT_API_URL = os.getenv("WILDFIRE_API_URL", "").rstrip("/")
DEFAULT_TIMEOUT = float(os.getenv("WILDFIRE_API_TIMEOUT_SECONDS", "120"))


@st.cache_data(ttl=120)
def fetch_model_info(api_base_url: str) -> dict:
    response = httpx.get(f"{api_base_url}/model-info", timeout=20.0)
    response.raise_for_status()
    return response.json()


def build_temporal_context(
    temperature_c: float,
    humidity_pct: float,
    wind_speed_kph: float,
    drought_index: float,
    vegetation_dryness: float,
) -> list[dict[str, float]]:
    return [
        {
            "temperature_c": round(temperature_c - 4.0, 2),
            "humidity_pct": round(min(humidity_pct + 8.0, 100.0), 2),
            "wind_speed_kph": round(max(wind_speed_kph - 6.0, 0.0), 2),
            "drought_index": round(max(drought_index - 55.0, 0.0), 2),
            "vegetation_dryness": round(max(vegetation_dryness - 0.10, 0.0), 3),
        },
        {
            "temperature_c": round(temperature_c - 2.0, 2),
            "humidity_pct": round(min(humidity_pct + 4.0, 100.0), 2),
            "wind_speed_kph": round(max(wind_speed_kph - 3.0, 0.0), 2),
            "drought_index": round(max(drought_index - 30.0, 0.0), 2),
            "vegetation_dryness": round(max(vegetation_dryness - 0.05, 0.0), 3),
        },
        {
            "temperature_c": round(temperature_c, 2),
            "humidity_pct": round(humidity_pct, 2),
            "wind_speed_kph": round(wind_speed_kph, 2),
            "drought_index": round(drought_index, 2),
            "vegetation_dryness": round(vegetation_dryness, 3),
        },
    ]


def call_predict(
    api_base_url: str,
    image_bytes: bytes,
    image_name: str,
    content_type: str,
    form_data: dict[str, str],
) -> dict:
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(
            f"{api_base_url}/predict",
            files={"file": (image_name, image_bytes, content_type)},
            data=form_data,
        )
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Wildfire Risk Intelligence", page_icon="🔥", layout="wide")

st.title("Wildfire Risk Intelligence")
st.caption(
    "Public wildfire risk demo running as a Hugging Face Docker Space "
    "and calling a FastAPI backend."
)

with st.sidebar:
    st.subheader("Backend")
    api_base_url = st.text_input(
        "FastAPI base URL",
        value=DEFAULT_API_URL,
        placeholder="https://wildfire-risk-api.onrender.com",
    ).rstrip("/")
    st.caption("Point this at the deployed `/predict` API.")

    st.subheader("Environmental Context")
    temperature_c = st.number_input(
        "Temperature (C)",
        min_value=-20.0,
        max_value=60.0,
        value=34.0,
    )
    humidity_pct = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=22.0)
    wind_speed_kph = st.number_input(
        "Wind speed (km/h)",
        min_value=0.0,
        max_value=150.0,
        value=24.0,
    )
    drought_index = st.number_input("Drought index", min_value=0.0, max_value=800.0, value=410.0)
    vegetation_dryness = st.slider("Vegetation dryness", min_value=0.0, max_value=1.0, value=0.78)
    days_since_rain = st.number_input("Days since rain", min_value=0.0, max_value=90.0, value=11.0)
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.0)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-121.0)
    include_temporal = st.checkbox("Send temporal context", value=True)

uploaded_file = st.file_uploader("Upload wildfire image", type=["jpg", "jpeg", "png"])

if not api_base_url:
    st.warning("Set `WILDFIRE_API_URL` or paste the deployed backend URL in the sidebar.")
    st.stop()

with st.expander("Live model info", expanded=False):
    try:
        st.json(fetch_model_info(api_base_url))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach backend: {exc}")

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Invalid image: {exc}")
        st.stop()

    temporal_context = None
    if include_temporal:
        temporal_context = build_temporal_context(
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            wind_speed_kph=wind_speed_kph,
            drought_index=drought_index,
            vegetation_dryness=vegetation_dryness,
        )

    payload = {
        "temperature_c": str(temperature_c),
        "humidity_pct": str(humidity_pct),
        "wind_speed_kph": str(wind_speed_kph),
        "drought_index": str(drought_index),
        "vegetation_dryness": str(vegetation_dryness),
        "days_since_rain": str(days_since_rain),
        "latitude": str(latitude),
        "longitude": str(longitude),
        "include_explainability": "true",
    }
    if temporal_context is not None:
        payload["temporal_context_json"] = json.dumps(temporal_context)

    with st.spinner("Running multimodal inference..."):
        try:
            prediction = call_predict(
                api_base_url=api_base_url,
                image_bytes=image_bytes,
                image_name=uploaded_file.name,
                content_type=uploaded_file.type or "image/jpeg",
                form_data=payload,
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            st.error(f"Prediction failed with status {exc.response.status_code}: {detail}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
            st.stop()

    image_col, explain_col = st.columns(2)
    with image_col:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with explain_col:
        overlay_base64 = prediction.get("explainability", {}).get("overlay_base64")
        if overlay_base64:
            overlay = Image.open(BytesIO(base64.b64decode(overlay_base64))).convert("RGB")
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted class", prediction["class_name"])
    metric_col2.metric("Image confidence", f'{prediction["confidence"]:.2%}')
    metric_col3.metric("Overall risk", f'{prediction["overall_risk_score"]:.2%}')

    st.markdown("### Risk summary")
    st.write(f'Risk level: `{prediction["risk_level"]}`')
    st.write(prediction["recommended_action"])

    summary = prediction.get("explainability", {}).get("summary")
    if summary:
        st.caption(summary)

    st.markdown("### Probability breakdown")
    st.json(prediction["probabilities"])

    context_risk_score = prediction.get("context_risk_score")
    if context_risk_score is not None:
        st.markdown("### Contextual drivers")
        st.write(f"Context risk score: `{context_risk_score:.2%}`")
        for contributor in prediction.get("top_contributors", []):
            st.write(f'- `{contributor["factor"]}`: {contributor["rationale"]}')
