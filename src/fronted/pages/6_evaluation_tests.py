import logging
import time
import streamlit as st
import cv2
from backend.utils import (format_time, create_images_pool_from_video, create_mosaic_from_video_frames_multi_thread,
                           get_llm_inference)
from fronted.utils import create_title_page, create_start_analysis_button, video_connector, \
    convert_minutes_seconds_to_frames, \
    create_minute_seconds_button, create_mosaic_plot_dimension_icon, \
    show_video, create_llm_models_menu, create_batch_size_and_frame_to_skip_frequency_buttons, \
    visualize_current_frame_inference_frame_and_mosaic_plots_for_inference, get_true_label_prediction_and_tokens_per_second, \
    create_thread_label_buttons, save_and_display_classification_report_to_csv
from src.backend.prompts import HAZARD_DETECTION_GRID_PLOT_PROMPT, SIMPLE_PROMPT_SMALL_MODELS

MAX_NUM_TOKENS = 300
MIN_RESPONSE_TIME_FAKE_MODEL = 1.0
MAX_RESPONSE_TIME_FAKE_MODEL = 2.0
# PROMPT = HAZARD_DETECTION_PROMPT
#PROMPT = SIMPLE_PROMPT_SMALL_MODELS
PROMPT = HAZARD_DETECTION_GRID_PLOT_PROMPT

logging.basicConfig(level=logging.INFO)


def llm_inference_and_visualization():
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
        minutes_initial_frame, seconds_initial_frame = (
            create_minute_seconds_button(title=f"#### **Choose an initial frame**"))
        initial_frame = convert_minutes_seconds_to_frames(minutes=minutes_initial_frame, seconds=seconds_initial_frame)

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

        # TESTING
        # Thread label buttons
        create_thread_label_buttons()

        if start_clicked:

            video, total_frames, fps = video_connector(path=st.session_state.video_path)

            # ToDo: In the real app code below should come from the camera.
            frame_pool = create_images_pool_from_video(st.session_state.video_path)

            current_frame_after_llm_response_old = initial_frame
            current_frame_after_llm_response = initial_frame
            inference_time_previous_iter = 0

            video.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

            labels = []
            predictions = []
            llm_latencies = []
            overall_latencies = []
            tokens_per_second = []

            total_frames = st.session_state.video_metadata['total_frames']

            while video.isOpened():

                ret, frame = video.read()
                # if not ret:
                #    break

                current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                progress_placeholder.write(
                    f"Frame {current_frame}/{total_frames} "
                    f"({format_time(current_frame)}/{format_time(total_frames / fps)})"
                )

                if current_frame_after_llm_response == initial_frame:
                    current_frame_after_llm_response = initial_frame + 1
                else:
                    current_frame_after_llm_response += int(inference_time_previous_iter * fps)
                    current_frame_after_llm_response = min(current_frame_after_llm_response, total_frames)

                # ToDo: In the real app, line below should come from the camera.
                frames_collected = frame_pool[current_frame_after_llm_response_old:current_frame_after_llm_response]

                # Preprocess Images
                start_time = time.time()

                mosaic_chunks = create_mosaic_from_video_frames_multi_thread(
                    frames=frames_collected,
                    start_index_subplot=current_frame_after_llm_response_old,  # previous_inference_frame_idx,
                    n_cols=mosaic_dimension,
                    n_rows=mosaic_dimension,
                    batch_size=None if not batch_size.isdigit() else int(batch_size),
                    frames_to_skip_frequency=frame_to_skip_freq
                )

                # LLM request
                llm_output = get_llm_inference(
                    model_name=selected_model_name,
                    frames=mosaic_chunks,
                    batch_mode=True,
                    prompt=PROMPT,
                    min_response_time_fake_model=MIN_RESPONSE_TIME_FAKE_MODEL,
                    max_response_time_fake_model=MAX_RESPONSE_TIME_FAKE_MODEL,
                    max_num_tokens=MAX_NUM_TOKENS
                )

                end_time = time.time()
                inference_time_previous_iter = end_time - start_time
                overall_latencies.append(inference_time_previous_iter)
                llm_latencies.append(llm_output[0].inference_time)

                real_frame_after_llm_response = min(int(current_frame_after_llm_response +
                                                    inference_time_previous_iter * fps), total_frames)


                # Streamlit results visualization
                visualize_current_frame_inference_frame_and_mosaic_plots_for_inference(
                    video=video,
                    current_frame_idx=real_frame_after_llm_response,
                    old_inference_frame_idx=current_frame_after_llm_response_old,
                    current_inference_frame_idx=current_frame_after_llm_response,
                    mosaic_chunks=mosaic_chunks,
                    llm_outputs=llm_output,
                    overall_inference_time=inference_time_previous_iter,
                    is_video_finished=not ret
                )

                # TESTING
                if st.session_state.test_evaluation:
                    label_prediction = get_true_label_prediction_and_tokens_per_second(
                        current_frame_idx=current_frame_after_llm_response,
                        old_inference_frame_idx=current_frame_after_llm_response_old,
                        llm_outputs=llm_output,
                        thread_frame_labels=st.session_state.initial_final_threat_frames_labels
                    )

                    if label_prediction is not None:
                        labels.append(label_prediction[0])
                        predictions.append(label_prediction[1])

                if not ret:
                    break

                # Get real image after LLM+Preprocessing response.
                update_video_frame = min(int(current_frame_after_llm_response + inference_time_previous_iter * fps),
                                         total_frames)
                video.set(cv2.CAP_PROP_POS_FRAMES, update_video_frame)

                current_frame_after_llm_response_old = current_frame_after_llm_response

            video.release()

            current_frame_placeholder.empty()
            progress_placeholder.empty()
            current_analysis_placeholder.empty()
            st.markdown("**Analysis Complete**")

            # TESTING
            if st.session_state.test_evaluation:
                st.markdown(f"## **EVALUATION TESTS**")
                #st.markdown(f"**Labels**: {labels}")
                #st.markdown(f"**Predictions**: {predictions}")
                if len(labels) > 0:
                    save_and_display_classification_report_to_csv(
                        true_labels=labels,
                        predicted_labels=predictions,
                        llm_latencies=llm_latencies,
                        overall_latencies=overall_latencies
                    )
                else:
                    st.markdown(f"## **Test skipped as no labels found. Please add the labels through the UI on "
                                f"the left.**")
                ## Here compute --> F1, Accuracy, Precision and Recall --> What is the important metric for us?


def main():
    llm_inference_and_visualization()


if __name__ == "__main__":
    main()
