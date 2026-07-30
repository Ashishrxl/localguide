import os
import streamlit as st
from google import genai
from google.genai import types

# Set up page configuration
st.set_page_config(
    page_title="Gemini Travel & Search Assistant",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Travel & Search Assistant")
st.write("Powered by Google GenAI SDK with Search & Maps Grounding")

# Retrieve API Key from Streamlit Secrets or Environment Variable
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY! Please configure it in your Streamlit secrets.", icon="🚨")
    st.stop()

# Initialize the GenAI Client
client = genai.Client(api_key=api_key)

# Input form for user query
user_query = st.text_input("Enter your query:", placeholder="e.g., Latest news about space exploration, or best coffee shops in Tokyo")

if st.button("Search / Generate", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        st.subheader("Results:")
        
        # Configure model content
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_query),
                ],
            ),
        ]

        # Enable both Google Search and Google Maps grounding
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(google_maps=types.GoogleMaps()),
        ]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="MINIMAL",
            ),
            tools=tools,
        )

        # Generator function to stream tokens directly into Streamlit
        def stream_response():
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=generate_content_config,
            )
            
            for chunk in response_stream:
                # Stream main response text
                if chunk.text:
                    yield chunk.text

        # Render response in real-time
        st.write_stream(stream_response)
      
