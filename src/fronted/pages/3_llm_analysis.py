import streamlit as st
import cv2
import os

from backend.utils import format_time, process_frame
from fronted.utils import display_analysis
from src.backend.prompts import HAZARD_DETECTION_PROMPT
from src.backend.model_providers import ModelProvider, get_model_provider


# Main app
st.title("Hazard Detection Analysis")

# Initialize basic session state
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

# Check for API key
if not os.getenv('OPENAI_API_KEY'):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

if st.session_state.video_file is None:
    st.info("Please upload a video file from the home page")
else:
    # Single control button at the top
    start_clicked = st.button("▶️ Start Analysis", 
                            use_container_width=True,
                            type="secondary")
    
    # Provider selection
    provider = st.selectbox(
        "Select Provider",
        options=[provider.value for provider in ModelProvider],
        index=0
    )
    
    # Initialize appropriate provider
    if provider == ModelProvider.OPENAI.value:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.stop()
        model_provider = get_model_provider(ModelProvider.OPENAI, api_key)
    else:  # Fireworks
        api_key = os.getenv('FIREWORKS_API_KEY')
        if not api_key:
            st.error("Fireworks API key not found. Please set the FIREWORKS_API_KEY environment variable.")
            st.stop()
        model_provider = get_model_provider(ModelProvider.FIREWORKS, api_key)
    
    # Model selection based on provider
    model_options = model_provider.available_models
    selected_model_name = st.selectbox("Select Model", options=list(model_options.keys()), index=0)
    model_provider.model_name = model_options[selected_model_name]
    
    # Create placeholders for current analysis
    current_frame_placeholder = st.empty()
    progress_placeholder = st.empty()
    current_analysis_placeholder = st.empty()
    
    # Show analysis status
    st.markdown("**Status:** Waiting to Start")
    
    # Container for historical analyses
    st.markdown("### Analysis History")
    history_container = st.container()
    
    if start_clicked:
        video = cv2.VideoCapture(st.session_state.video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = current_frame / fps
            
            # Update progress
            progress_placeholder.write(
                f"Frame {current_frame}/{total_frames} "
                f"({format_time(current_time)}/{format_time(total_frames / fps)})"
            )
            
            # Save current position
            current_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Analyze frame first
            analysis, inference_time = model_provider.analyze_frame(frame, HAZARD_DETECTION_PROMPT)
            
            if analysis:
                # Show three columns in history
                with history_container:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        # Calculate and show future frame
                        frames_ahead = int(inference_time * fps)
                        future_frame_pos = min(current_frame + frames_ahead, total_frames - 1)
                        video.set(cv2.CAP_PROP_POS_FRAMES, future_frame_pos)
                        ret, future_frame = video.read()
                        
                        if ret:
                            st.markdown("**Current Video Frame**")
                            future_time = future_frame_pos / fps
                            st.image(
                                process_frame(future_frame),
                                caption=f"Frame {future_frame_pos} ({format_time(future_time)})"
                            )
                            st.markdown(f"*Latency gap: {inference_time:.1f}s ({frames_ahead} frames)*")
                    
                    # Return to original position
                    video.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    
                    with col2:
                        st.markdown("**Frame Being Analyzed**")
                        st.image(
                            process_frame(frame),
                            caption=f"Frame {current_frame} ({format_time(current_time)})"
                        )
                    
                    with col3:
                        display_analysis(analysis, inference_time)
                    
                    st.divider()
            
            # Calculate next frame based on inference time
            frames_to_skip = int(inference_time * fps)
            next_frame = current_frame + frames_to_skip
            video.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        
        video.release()
        
        # Clear current analysis displays when done
        current_frame_placeholder.empty()
        progress_placeholder.empty()
        current_analysis_placeholder.empty()
        st.markdown("**Analysis Complete**")

