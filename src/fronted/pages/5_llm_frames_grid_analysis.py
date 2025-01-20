import logging
import time
import streamlit as st
import cv2
from src.backend.utils import format_time, create_images_pool_from_video, create_mosaic_from_video_frames, \
    create_mosaic_from_video_frames_multi_thread, get_llm_inference
from src.fronted.utils import create_title_page, show_video, create_llm_models_menu, \
    create_minute_seconds_button, convert_minutes_seconds_to_frames, create_mosaic_plot_dimension_icon, \
    create_batch_size_and_frame_to_skip_frequency_buttons, create_start_analysis_button, video_connector, \
    visualize_current_frame_inference_frame_and_mosaic_plots_for_inference
from src.backend.prompts import HAZARD_DETECTION_PROMPT, SIMPLE_PROMPT_SMALL_MODELS, HAZARD_DETECTION_GRID_PLOT_PROMPT

MAX_NUM_TOKENS = 200
FAKE_MODEL_NAME = "Fake_Model"
MIN_RESPONSE_TIME_FAKE_MODEL = 1.5
MAX_RESPONSE_TIME_FAKE_MODEL = 1.5
#PROMPT = HAZARD_DETECTION_PROMPT
#PROMPT = SIMPLE_PROMPT_SMALL_MODELS
PROMPT = HAZARD_DETECTION_GRID_PLOT_PROMPT

logging.basicConfig(level=logging.INFO)


def llm_analysis_page():

    create_title_page(title="Hazard Detection Analysis")

    if 'analyzing' not in st.session_state:
        st.session_state.analyzing = False

    if st.session_state.video_file is None:
        st.info("Please upload a video file from the home page")
    else:

        show_video()

        # LLM Model
        selected_model_name = create_llm_models_menu()

        # Choose initial frame
        minutes, seconds = create_minute_seconds_button(title=f"#### **Choose an initial frame**")
        initial_frame = convert_minutes_seconds_to_frames(minutes=minutes, seconds=seconds)

        # Mosaic Plot dimension
        mosaic_dimension = create_mosaic_plot_dimension_icon()

        batch_size, frame_to_skip_freq = create_batch_size_and_frame_to_skip_frequency_buttons()

        # Start Analysis Button
        start_clicked = create_start_analysis_button()

        current_frame_placeholder = st.empty()
        progress_placeholder = st.empty()
        current_analysis_placeholder = st.empty()

        st.markdown("**Status:** Waiting to Start")
        st.markdown("### Analysis History")

        if start_clicked:

            video, total_frames, fps = video_connector(path=st.session_state.video_path)

            # ToDo: In the real app code below should come from the camera.
            frame_pool = create_images_pool_from_video(st.session_state.video_path)

            frames_collected = []
            current_frame_after_llm_response_old = initial_frame
            current_frame_after_llm_response = initial_frame
            inference_time_previous_iter = 0

            video.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

            while video.isOpened():

                ret, frame = video.read()
                if not ret:
                    break

                current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                progress_placeholder.write(
                    f"Frame {current_frame}/{total_frames} "
                    f"({format_time(current_frame)}/{format_time(total_frames / fps)})"
                )

                if current_frame_after_llm_response == initial_frame:
                    current_frame_after_llm_response = initial_frame + 1
                else:
                    current_frame_after_llm_response += int(inference_time_previous_iter*fps)

                # ToDo: In the real app, line below should come from the camera.
                frames_collected = frame_pool[current_frame_after_llm_response_old:current_frame_after_llm_response]

                # Preprocess Images
                start_time = time.time()

                mosaic_chunks = create_mosaic_from_video_frames_multi_thread(
                    frames=frames_collected,
                    start_index_subplot=current_frame_after_llm_response_old,  # previous_inference_frame_idx,
                    n_cols=mosaic_dimension,
                    n_rows=mosaic_dimension,
                    batch_size= None if not batch_size.isdigit() else int(batch_size),
                    frames_to_skip_frequency=frame_to_skip_freq
                )

                # LLM request
                llm_output = get_llm_inference(
                    model_name=selected_model_name,
                    frames=mosaic_chunks,
                    batch_mode=True,
                    prompt=PROMPT,
                    fake_model=True if selected_model_name == FAKE_MODEL_NAME else False,
                    min_response_time_fake_model=MIN_RESPONSE_TIME_FAKE_MODEL,
                    max_response_time_fake_model=MAX_RESPONSE_TIME_FAKE_MODEL,
                    max_num_tokens=MAX_NUM_TOKENS
                )

                # Streamlit results visualization

                end_time = time.time()

                # ToDo: In the real app code below should come from the camera.
                # inference_time_previous_iter = llm_output.inference_time
                inference_time_previous_iter = end_time - start_time

                real_frame_after_llm_response = int(current_frame_after_llm_response +
                                                    inference_time_previous_iter * fps)

                visualize_current_frame_inference_frame_and_mosaic_plots_for_inference(
                    video=video,
                    current_frame_idx=real_frame_after_llm_response,
                    current_inference_frame_idx=current_frame_after_llm_response,
                    mosaic_chunks=mosaic_chunks,
                    llm_outputs=llm_output,
                    overall_inference_time=inference_time_previous_iter
                )

                # Get real image after LLM+Preprocessing response.
                video.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_after_llm_response +
                                                       inference_time_previous_iter*fps))

                current_frame_after_llm_response_old = current_frame_after_llm_response

            video.release()

            current_frame_placeholder.empty()
            progress_placeholder.empty()
            current_analysis_placeholder.empty()
            st.markdown("**Analysis Complete**")


def main():
    llm_analysis_page()


if __name__ == "__main__":
    main()
