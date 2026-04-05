---
title: Wildfire Risk Intelligence
emoji: "🔥"
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-basic
short_description: Streamlit frontend for multimodal wildfire risk inference.
---

# Wildfire Risk Intelligence

This Docker Space hosts the public Streamlit frontend for the wildfire risk prediction system.

Required Space variable or secret:

- `WILDFIRE_API_URL`: public base URL of the FastAPI backend, for example `https://wildfire-risk-api.onrender.com`

Optional variable:

- `WILDFIRE_API_TIMEOUT_SECONDS`: request timeout for prediction calls, default `120`
