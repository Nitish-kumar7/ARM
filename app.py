import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Pediatric Ultrasound Analysis",
    page_icon="🏥",
    layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # Inject custom CSS
    local_css("style.css")

    # Sidebar content
    st.sidebar.title("About")
    st.sidebar.markdown("""
    Pediatric Ultrasound Analysis helps you screen pediatric ultrasound images for potential issues using AI.
    """)

    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload a pediatric ultrasound image (JPG, JPEG, PNG)
    2. The model will classify it as Normal or ARM (placeholder)
    3. A heatmap will show the areas the model focused on (placeholder)
    """)
    # Added Disclaimer section
    st.sidebar.title("Disclaimer")
    st.sidebar.markdown("""
    *   This tool is for educational/screening purposes only
    *   Always consult a healthcare professional for medical decisions
    """)
    # Added Contact & Help section
    st.sidebar.title("Contact & Help")
    st.sidebar.markdown("""
    *   [GitHub](https://github.com/your_repo) (Replace with your GitHub link)
    *   [Streamlit Docs](https://docs.streamlit.io/)
    *   Email: support@example.com (Replace with your email)
    """)

    # Main content area
    st.title("Pediatric Ultrasound Analysis")
    st.markdown("Upload a pediatric ultrasound image to detect potential issues using AI.")

    # File uploader
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Placeholder sections for results and Grad-CAM
        st.subheader("Analysis Results (Placeholder)")
        st.write("Classification Result: Not Available (Model integration pending)")
        st.write("Confidence: Not Available (Model integration pending)")

        st.subheader("Grad-CAM Visualization (Placeholder)")
        st.image(image, caption="Grad-CAM pending", use_column_width=True)

if __name__ == "__main__":
    main()
