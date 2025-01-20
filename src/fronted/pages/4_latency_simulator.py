import streamlit as st
import cv2
import time
import numpy as np
import os
from src.backend.prompts import HAZARD_DETECTION_PROMPT
from src.backend.model_providers import ModelProvider, get_model_provider

def process_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def format_time(seconds):
    return f"{int(seconds/60):02d}:{int(seconds%60):02d}"

def simulate_llm_response(latency):
    """Simulate an LLM response with specified latency"""
    time.sleep(latency)
    # Generate a random response
    hazard = np.random.choice([True, False], p=[0.3, 0.7])  # 30% chance of hazard
    hazard_type = "none" if not hazard else np.random.choice(["theft", "robbery"])
    reasoning = (
        "Suspicious individual approaching vehicle" if hazard 
        else "Normal pedestrian activity in view"
    )
    return {
        "hazard_detected": hazard,
        "hazard_type": hazard_type,
        "reasoning": reasoning
    }

st.title("Latency Testing Simulator")

if st.session_state.video_file is None:
    st.info("Please upload a video file from the home page")
else:
    # Get video metadata first for time range
    video = cv2.VideoCapture(st.session_state.video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    video.release()
    
    # Video info
    st.write(f"Video duration: {format_time(duration)} ({duration:.1f} seconds)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Latency slider
        latency = st.slider(
            "Latency (seconds)",
            min_value=0.1,
            max_value=4.0,
            value=2.0,
            step=0.1,
            help="Simulate different LLM response times"
        )
    
    with col2:
        # Start time selector
        start_time = st.slider(
            "Start Time",
            min_value=0.0,
            max_value=float(duration),
            value=0.0,
            step=1.0,
            format="%d seconds",
            help="Select when to start analysis"
        )
    
    with col3:
        # End time selector
        end_time = st.slider(
            "End Time",
            min_value=0.0,
            max_value=float(duration),
            value=float(duration),
            step=1.0,
            format="%d seconds",
            help="Select when to end analysis"
        )
    
    # Validate time range
    if start_time >= end_time:
        st.error("Start time must be before end time")
        st.stop()
    
    st.write(f"Analysis range: {format_time(start_time)} - {format_time(end_time)} "
             f"(Duration: {format_time(end_time - start_time)})")
    
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
    
    # Single control button
    start_clicked = st.button("▶️ Start Latency Test", 
                            use_container_width=True,
                            type="secondary")
    
    # Container for frame analyses
    st.markdown("### Frame Analysis Results")
    results_container = st.container()
    
    if start_clicked:
        video = cv2.VideoCapture(st.session_state.video_path)
        
        # Set starting position
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        with results_container:
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                
                current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = current_frame / fps
                
                # Check if we've reached the end time
                if current_frame >= end_frame:
                    break
                
                # Process current frame
                col1, col2, col3 = st.columns([1, 1, 1])
                
                # Save current position to return to it later
                current_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
                
                with col1:
                    # Calculate and show the frame that would be current after latency
                    frames_ahead = int(latency * fps)
                    future_frame_pos = min(current_frame + frames_ahead, end_frame)
                    video.set(cv2.CAP_PROP_POS_FRAMES, future_frame_pos)
                    ret, future_frame = video.read()
                    
                    if ret:
                        st.markdown("**Current Video Frame**")
                        future_time = future_frame_pos / fps
                        st.image(
                            process_frame(future_frame),
                            caption=f"Frame {future_frame_pos} ({format_time(future_time)})"
                        )
                        st.markdown(f"*Latency gap: {latency:.1f}s ({frames_ahead} frames)*")
                    
                    # Return to the original position
                    video.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

                with col2:
                    st.markdown("**Frame Being Analyzed**")
                    st.image(
                        process_frame(frame),
                        caption=f"Frame {current_frame} ({format_time(current_time)})"
                    )
                
                with col3:
                    # Get real or simulated analysis based on model selection
                    start_time = time.time()
                    
                    if provider == "Simulation":
                        analysis = simulate_llm_response(latency)
                    else:
                        analysis, _ = model_provider.analyze_frame(frame, HAZARD_DETECTION_PROMPT)
                    
                    inference_time = time.time() - start_time
                    
                    # Display analysis
                    bg_color = "#ffcdd2" if analysis["hazard_detected"] else "#c8e6c9"
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 5px; background-color: {bg_color}; color: black;">
                            <p><strong>Status:</strong> {'⚠️ HAZARD DETECTED' if analysis["hazard_detected"] else '✅ NO HAZARD DETECTED'}</p>
                            <p><strong>Type:</strong> {analysis["hazard_type"]}</p>
                            <p><strong>Reasoning:</strong> {analysis["reasoning"]}</p>
                            <p><em>Analysis took {inference_time:.1f}s</em></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.divider()
                
                # Skip frames based on latency but don't exceed end time

                #latency = latency if provider == "Simulation" else inference_time

                frames_to_skip = int(latency * fps)
                next_frame = min(current_frame + frames_to_skip, end_frame)
                video.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
            
            video.release()
            st.markdown("**Analysis Complete**") 