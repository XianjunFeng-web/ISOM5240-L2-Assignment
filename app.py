from transformers import pipeline
import streamlit as st
from PIL import Image
import time

st.set_page_config(page_title="Story telling application", page_icon="🦜")

# App title
st.title("🌟 Storytelling Application 🌟")

# Write some text
st.write("🤓 Welcome to storytelling app! 🖼️")
st.write("☀️ Sweetie, it is wonderful story time~ Let's enjoy the story! ✨❤️")

# --- Functions ---
@st.cache_resource # This makes the model load faster after the first time
def get_models():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_gen = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    audio_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    return captioner, story_gen，audio_pipe

captioner, story_gen = get_models()

def img2text(file):
    image = Image.open(file)
    result = captioner(image)
    return result[0]['generated_text']

def text2story(text):
    prompt = f"Write a short, creative story based on this scene: {text}. The story begins: "
    story = story_gen(prompt, max_length=150, do_sample=True, temperature=0.7)
    return story[0]['generated_text']

def story2audio(story_text):
    # This turns the story text into audio waves
    audio_output = audio_pipe(story_text)
    return audio_output

    
    

# --- Main Part  ---
st.header("🤩 Pick up an image to start")
uploaded_image = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process the image automatically once uploaded
    with st.spinner("Turning your image into a description..."):
        scenario = img2text(uploaded_image)
        st.write(f"**Description:** {scenario}")

    # Now show the button to generate the story
    if st.button("✨ Generate My Story ✨"):
        with st.spinner("Writing a magical story..."):
            story = text2story(scenario)
            st.write("---")
            st.subheader("📖 The Story")
            st.write(story)
else:
    st.info("Please upload an image to start the story.")

    # Now show the button to generate the story in audio
   if 'saved_story' in st.session_state:
        st.write("---")
        st.subheader("📖 The Story")
        st.write(st.session_state['saved_story'])
   # 3. If a story exists, show it and show the Audio Button
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

