import streamlit as st
from PIL import Image
import time

# App title
st.title("🌟 Storytelling Application 🌟")
st.set_page_config(page_title="Story telling application", page_icon="🦜")

# Write some text
st.write("🤓Welcome to storytelling app!🖼️")
st.write("☀️Sweetie，It is the wonderful story time~ ~Let's enjoy the story! ✨❤️")

# Function part
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(uploaded_image)[0]["generated_text"]
    return text
    
# Main part
   
# File uploader for image and audio
st.header("First, pick up an image and then start our story journey 🔐")

uploaded_image = st.file_uploader("upload image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    with st.spinner("Loading image..."):
        time.sleep(1)  # Simulate a delay
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

st.header("Click to turn your Image into Audio Story")

# Button interaction
if st.button("Click Me"):
    st.write("🎉 Turned into below Audio Story!")
   
    
#  Stage 1: Image to Text (Using the function)
    st.text('Turn Your Image to story in text...')
    scenario = img2text(uploaded_image.name)
    st.write(f"**Scenario:** {scenario}")




