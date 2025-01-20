import streamlit as st


def format_time(seconds):
    """Convert seconds to MM:SS format"""
    return f"{int(seconds/60):02d}:{int(seconds%60):02d}"


st.title("Video Player")

if st.session_state.video_file is None:
    st.info("Please upload a video file from the home page")
else:
    metadata = st.session_state.video_metadata
    
    # Video info
    st.write(f"Video duration: {format_time(metadata['duration'])} ({metadata['duration']:.1f} seconds)")
    st.write(f"Total frames: {metadata['total_frames']}")
    st.write(f"FPS: {metadata['fps']}")
    
    # Play video
    st.video(st.session_state.video_path)
