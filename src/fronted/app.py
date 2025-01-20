import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # Add this line to load the .env file

st.set_page_config(
    page_title="Video Frame Analyzer",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state for video
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'video_metadata' not in st.session_state:
    st.session_state.video_metadata = None

st.title("Video Frame Analyzer with GPT")

# Global video upload
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
if uploaded_file != st.session_state.video_file:
    st.session_state.video_file = uploaded_file
    if uploaded_file is not None:
        import tempfile
        import cv2
        
        # Save video file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        
        # Get video metadata
        video = cv2.VideoCapture(st.session_state.video_path)
        st.session_state.video_metadata = {
            'total_frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': video.get(cv2.CAP_PROP_FPS),
            'duration': int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / video.get(cv2.CAP_PROP_FPS)
        }
        video.release()
    else:
        st.session_state.video_path = None
        st.session_state.video_metadata = None

st.write("""
This application analyzes video frames using GPT. 
         
Navigate using the sidebar to:
- 📹 Video: Play the uploaded video
- ⚙️ Frame Analysis: Analyze video frames with configurable inference time
""")
