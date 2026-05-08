
from transformers import pipeline
import streamlit as st
from PIL import Image
import time
import torch

st.set_page_config(page_title="Story telling application", page_icon="🦜")

# App title
st.title("🌟 Storytelling Application 🌟")

st.write("🤓 Welcome to storytelling app! 🖼️")
st.write("☀️ Sweetie, it is wonderful story time~ Let's enjoy the story! ✨❤️")

# --- Functions ---
@st.cache_resource 
def get_models():
    # Loading all three AI models
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_gen = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    # Using a robust Text-to-Speech model
    audio_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    return captioner, story_gen, audio_pipe

captioner, story_gen, audio_pipe = get_models()

def img2text(file):
    image = Image.open(file)
    result = captioner(image)
    return result[0]['generated_text']

def text2story(text):
    prompt = f"Write a short, creative story based on this scene: {text}. The story begins: "
    story_output = story_gen(prompt, max_length=150, do_sample=True, temperature=0.7)
    return story_output[0]['generated_text']

def story2audio(story_text):
    # This turns the story text into audio waves
    audio_output = audio_pipe(story_text)
    return audio_output

# --- Main Part ---
st.header("🤩 Pick up an image to start")
uploaded_image = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # 1. Get Image Caption
    with st.spinner("Turning your image into a description..."):
        scenario = img2text(uploaded_image)
        st.write(f"**Description:** {scenario}")

    # 2. Story Generation Button
    if st.button("✨ Generate My Story ✨"):
        with st.spinner("Writing a magical story..."):
            generated_story = text2story(scenario)
            # Save the story into session state so it doesn't disappear
            st.session_state['saved_story'] = generated_story

    # 3. If a story exists, show it and show the Audio Button
    if 'saved_story' in st.session_state:
        st.write("---")
        st.subheader("📖 The Story")
        st.write(st.session_state['saved_story'])

        if st.button("🎧 Play Audio"):
            with st.spinner("Converting story to speech..."):
                audio_data = story2audio(st.session_state['saved_story'])
                
                # Extract the audio array and sample rate from the model output
                audio_array = audio_data["audio"]
                sample_rate = audio_data["sampling_rate"]
                
                # Display the audio player
                st.audio(audio_array, sample_rate=sample_rate)
else:
    st.info("Please upload an image to start the story.")
