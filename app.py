import streamlit as st

st.set_page_config(
    page_title="UCF-Crime Abnormal Behaviour Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 UCF-Crime Abnormal Behaviour Detection")
st.write("Upload an image to detect whether the scene is Normal or Abnormal.")

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔎 Analyze Image"):
        st.info("Model prediction will appear here.")
