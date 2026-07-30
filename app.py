import os
import streamlit as st
from google import genai
from google.genai import types

# Page config
st.set_page_config(
    page_title="Gemini Multimodal Assistant",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Travel & Search Assistant")
st.write("Ask via text or voice using Search & Maps grounding!")

# Retrieve API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY! Please configure it in your Streamlit secrets.", icon="🚨")
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)

# Input Section: Text + Audio
user_query = st.text_input("Text Prompt (Optional if recording audio):", placeholder="e.g., Explain what I just asked or search for places near me...")
audio_input = st.audio_input("Record your voice prompt:")

if st.button("Submit Query", type="primary"):
    if not user_query.strip() and not audio_input:
        st.warning("Please provide either a text prompt or record an audio message.")
    else:
        st.subheader("Results:")
        
        parts = []

        # 1. Attach Audio Byte Data if present
        if audio_input:
            audio_bytes = audio_input.read()
            parts.append(
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"  # st.audio_input records in wav/ogg format
                )
            )

        # 2. Attach Text Prompt if present
        if user_query.strip():
            parts.append(types.Part.from_text(text=user_query))
        elif audio_input:
            # Fallback text instruction if only audio is provided
            parts.append(
                types.Part.from_text(
                    text="Please listen to the attached audio and fulfill the user's request using search/maps grounding if needed."
                )
            )

        contents = [
            types.Content(
                role="user",
                parts=parts
            )
        ]

        # Enable Search and Maps grounding tools
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(google_maps=types.GoogleMaps()),
        ]

        generate_content_config = types.GenerateContentConfig(
            tools=tools,
        )

        def stream_response():
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=contents,
                config=generate_content_config,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        st.write_stream(stream_response)
