import os
import streamlit as st
from google import genai
from google.genai import types

# Set up page configuration
st.set_page_config(
    page_title="Gemini Multimodal Assistant",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS to force side-by-side layout on mobile screens
st.markdown("""
    <style>
    /* Prevent column stacking on mobile screens */
    div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        gap: 0.5rem !important;
        align-items: flex-end !important;
    }

    /* Force columns to retain proportions on mobile */
    div[data-testid="column"] {
        min-width: 0px !important;
    }

    /* Fix audio input widget padding inside small column */
    div[data-testid="stAudioInput"] {
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🗺️ Travel & Search Assistant")
st.write("Ask via text or voice using Search & Maps grounding!")

# Retrieve API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY! Please configure it in your Streamlit secrets.", icon="🚨")
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)

# Enforce bottom alignment natively across columns
col1, col2 = st.columns([5, 1], vertical_alignment="bottom")

with col1:
    user_query = st.text_input(
        "Ask Gemini", 
        placeholder="Type query or record...",
        label_visibility="collapsed"
    )

with col2:
    audio_input = st.audio_input("Record", label_visibility="collapsed")

if st.button("Submit Query", type="primary", use_container_width=True):
    if not user_query.strip() and not audio_input:
        st.warning("Please enter a text prompt or record audio.")
    else:
        st.subheader("Results:")
        
        parts = []

        # 1. Attach Audio Data if present
        if audio_input:
            audio_bytes = audio_input.read()
            parts.append(
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"
                )
            )

        # 2. Attach Text Prompt if present
        if user_query.strip():
            parts.append(types.Part.from_text(text=user_query))
        elif audio_input:
            parts.append(
                types.Part.from_text(
                    text="Please listen to the attached audio and fulfill the request using search or maps grounding if needed."
                )
            )

        contents = [
            types.Content(
                role="user",
                parts=parts
            )
        ]

        # Enable Grounding Tools
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
