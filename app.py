from transformers import pipeline
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
    # Function 1
    def img2text(file):
        # If you changed the model name inside here, that's fine!
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        image = Image.open(file)
        result = pipe(image)
        return result[0]['generated_text']
    
    #Function 2
    def text2story(text):
        # This creates a text-generation pipeline
        story_model = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
        
        # We ask the model to take your caption and write a short story
        prompt = f"Write a short, creative story based on this scene: {text}. The story begins: "
        story = story_model(prompt, max_length=150, do_sample=True, temperature=0.7)
        return story[0]['generated_text']
    
# Main part   
    # Step 1: Image uploader
        st.header("🤩Pick up an image and then start our story journey🎉")
        
        uploaded_image = st.file_uploader("upload image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            with st.spinner("Loading image..."):
                time.sleep(1)  # Simulate a delay
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # Optional: Tell the user to upload a file
            st.info("Please upload an image to start the story.")   
        
    #  Step2: Convert image to Text (Using the function)    
        st.header("🤩Click to turn your Image into Story in Text")
        
        #  Make a button to generate Text (Using the function) 
        if st.button("Click Me"):# Button interaction
            st.write("🎉 Turn the image into Story !")
               
        #  Convert image into Text (Using the function)
            st.text('Turn Your Image to story in text...')
        
        if uploaded_image is not None:
            #  Get the caption (The simple sentence)
            scenario = img2text(uploaded_image)
            st.write(f"**Caption:** {scenario}")
        
    # Step 3: Turn the caption into a story in paragraphs(use function)
        if st.button("Generate Story"):
            with st.spinner("Writing your story..."):
                story = text2story(scenario)
                st.write(f"**Full Story:**{story}")
             
    
    # Step 4: Click to convert text story for audio
    



