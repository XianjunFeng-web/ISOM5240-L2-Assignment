from transformers import pipeline
import streamlit as st
from PIL import Image
import torch

# 1. Page Configuration
st.set_page_config(page_title="Story telling application", page_icon="🦜")

# --- Functions ---

@st.cache_resource 
def get_models():
    """Function 1: Load all AI engines"""
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_gen = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    audio_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    return captioner, story_gen, audio_pipe

# Initialize models
captioner, story_gen, audio_pipe = get_models()

def img2text(file):
    """Function 2: Image to Caption"""
    image = Image.open(file)
    result = captioner(image)
    return result[0]['generated_text']

def text2story(text):
    """Function 3: Caption to Story"""
    prompt = f"Write a short, creative story based on this scene: {text}. The story begins: "
    story_output = story_gen(prompt, max_length=150, do_sample=True, temperature=0.7)
    return story_output[0]['generated_text']

def story2audio(story_text):
    """Function 4: Story to Speech"""
    audio_output = audio_pipe(story_text)
    return audio_output

# --- Main UI ---
st.title("🌟 Storytelling Application 🌟")
st.write("🤓 Welcome! Let's enjoy the story! ✨❤️")

uploaded_image = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(Image.open(uploaded_image), caption="Uploaded Image", use_container_width=True)
    
    # --- STEP 1: DESCRIBE ---
    st.header("Step 1: Describe Image")
    if st.button("Describe Image"):
        # We call the function and save the result to session_state
        st.session_state['scenario'] = img2text(uploaded_image)

    if 'scenario' in st.session_state:
        st.success(f"**Description:** {st.session_state['scenario']}")

        # --- STEP 2: STORY ---
        st.header("Step 2: Generate Story")
        if st.button("Generate Story"):
            # We call the function using the saved scenario
            st.session_state['story'] = text2story(st.session_state['scenario'])

    if 'story' in st.session_state:
        st.write("---")
        st.subheader("📖 The Story")
        st.write(st.session_state['story'])

        # --- STEP 3: AUDIO ---
        st.header("Step 3: Play Audio")
        if st.button("Play Audio"):
            with st.spinner("Generating voice..."):
                # We call the function using the saved story
                audio_data = story2audio(st.session_state['story'])
                st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])

else:
    st.info("Please upload an image to begin.")
