import streamlit as st
from PIL import Image
import time

# App title
st.title("🌟 Storytelling Application 🌟")

# Write some text
st.write("🤓Welcome to storytelling app!🖼️")
st.write("☀️Sweetie，It is the wonderful story time~ ~Let's enjoy the story! ✨❤️")

#
# File uploader for image and audio
uploaded_image = st.file_uploader("Identify today's fun by selecting a picture and then start our story journey",
                                  type=["jpg", "jpeg", "png"])


# Display image with spinner
if uploaded_image is not None:
    with st.spinner("Loading image..."):
        time.sleep(1)  # Simulate a delay
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Button interaction
if st.button("Click Me"):
    st.write("🎉 You clicked the button!")
