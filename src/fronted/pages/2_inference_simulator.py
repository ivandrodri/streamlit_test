import streamlit as st
import cv2
import time
import random


def mock_gpt_analysis(frame, min_time, max_time):
    """Mock GPT analysis of frame with varied responses"""
    inference_time = random.uniform(min_time, max_time)
    time.sleep(inference_time)
    
    responses = [
        "I can see a scene that appears to be {setting}. The overall mood seems {mood}.",
        "This frame captures {action}. The composition is {composition}.",
        "An interesting moment showing {subject}. Notable elements include {details}.",
        "The frame depicts {scene}. The lighting suggests {time_of_day}.",
    ]
    
    details = {
        'setting': ['indoors', 'in an urban environment', 'in nature', 'in a busy street'],
        'mood': ['calm and peaceful', 'energetic', 'mysterious', 'cheerful'],
        'action': ['people in motion', 'a quiet moment', 'an interaction between subjects', 'a landscape view'],
        'composition': ['balanced and symmetric', 'dynamic and engaging', 'minimalist', 'complex with many elements'],
        'subject': ['human activity', 'natural elements', 'architectural details', 'an interesting pattern'],
        'details': ['strong contrasts', 'subtle color variations', 'interesting textures', 'geometric patterns'],
        'scene': ['a slice of daily life', 'an atmospheric moment', 'a compelling narrative', 'a visual pattern'],
        'time_of_day': ['daytime', 'evening hours', 'early morning', 'artificial lighting']
    }
    
    template = random.choice(responses)
    response = template.format(**{k: random.choice(v) for k, v in details.items()})
    return response, inference_time

def process_frame(frame):
    """Convert frame to format suitable for display"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    return f"{int(seconds/60):02d}:{int(seconds%60):02d}"

st.title("Frame Analysis Simulator")

if st.session_state.video_file is None:
    st.info("Please upload a video file from the home page")
else:
    # Create inference time options (100ms to 5s in 100ms increments)
    time_options = [round(t/10, 1) for t in range(1, 51)]  # 0.1 to 5.0 seconds

    col1, col2 = st.columns(2)
    with col1:
        min_inference = st.selectbox(
            "Minimum inference time (seconds)",
            time_options,
            index=19  # Default to 2.0s
        )
    with col2:
        max_inference = st.selectbox(
            "Maximum inference time (seconds)",
            time_options,
            index=39  # Default to 4.0s
        )

    if min_inference > max_inference:
        st.error("Minimum inference time cannot be greater than maximum inference time")
        st.stop()

    st.write(f"Selected inference time range: {min_inference}s - {max_inference}s")
    
    metadata = st.session_state.video_metadata
    
    # Video info
    st.write(f"Video duration: {format_time(metadata['duration'])} ({metadata['duration']:.1f} seconds)")
    st.write(f"Total frames: {metadata['total_frames']}")
    
    # Create placeholders for frame analysis
    frame_placeholder = st.empty()
    progress_placeholder = st.empty()
    analysis_placeholder = st.empty()
    
    if st.button("Start Frame Analysis"):
        video = cv2.VideoCapture(st.session_state.video_path)
        current_frame = 0
        
        while video.isOpened() and current_frame < metadata['total_frames']:
            # Set video to current frame
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = video.read()
            if not ret:
                break
            
            current_time = current_frame / metadata['fps']
            
            # Update progress information
            progress_placeholder.write(
                f"Processing frame {current_frame}/{metadata['total_frames']} "
                f"({format_time(current_time)}/{format_time(metadata['duration'])})"
            )
            
            # Update current frame
            frame_placeholder.image(process_frame(frame))
            
            # Get analysis and inference time
            analysis, inference_time = mock_gpt_analysis(frame, min_inference, max_inference)
            analysis_placeholder.write(
                f"{analysis} (Inference time: {inference_time:.1f}s)"
            )
            
            # Calculate next frame based on inference time
            frames_to_skip = int(inference_time * metadata['fps'])
            current_frame += frames_to_skip
        
        video.release() 