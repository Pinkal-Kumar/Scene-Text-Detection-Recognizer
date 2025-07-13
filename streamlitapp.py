import streamlit as st
import requests
from PIL import Image

BACKEND_URL = "http://localhost:8000/infer/"

st.set_page_config(page_title="Scene Text Detection", layout="centered")
st.title("📷 Scene Text Detection & Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Inference"):
        with st.spinner("Sending to backend..."):
            try:
                # This is the working fix — send bytes directly
                files = {
                    "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                response = requests.post(BACKEND_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Inference successful!")
                    st.text_area("📝 Recognized Text", data["texts"], height=150)
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"❌ Request failed: {e}")
