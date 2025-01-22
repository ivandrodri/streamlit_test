import json
import logging
import os
from datetime import datetime
from typing import Tuple, List

import cv2
import numpy as np
import streamlit as st

from src.backend.factories.llm_factory import LlmModelFactory, LlmOutput
from src.backend.utils import process_frame, format_time

logging.basicConfig(level=logging.INFO)


def display_analysis(
        analysis: str,
        llm_inference_time: float = 0.0,
        overall_inference_time: float = 0.0,
        llm_cost: float = 0.0
):
    """Display analysis with appropriate color coding"""
    if not analysis:
        return

    try:
        # Try to parse as JSON if it's a string
        if isinstance(analysis, str):
            # Look for JSON pattern in the string
            import re
            json_match = re.search(r'\{.*\}', analysis)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # If JSON parsing fails, display as raw text with yellow background
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 5px; background-color: #fff3cd; color: black;">
                            <p><strong>⚠️ Unparseable JSON:</strong></p>
                            <p><code>{analysis}</code></p>
                            <p><em><strong>LLM inference time </strong> {llm_inference_time:.1f}s</em></p>
                            <p><em><strong>Latency (mosaic plot preprocessing + LLM inference time)</strong> {overall_inference_time:.1f}s</em></p>
                            <p><em><strong>Request cost</strong> {llm_cost:.6f}s</em></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    return
            else:
                # No JSON found, display as raw text with blue background
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 5px; background-color: #cce5ff; color: black;">
                        <p><strong>ℹ️ Raw Text:</strong></p>
                        <p>{analysis}</p>
                        <p><em><strong>LLM inference time </strong> {llm_inference_time:.1f}s</em></p>
                        <p><em><strong>Latency (mosaic plot preprocessing + LLM inference time)</strong> {overall_inference_time:.1f}s</em></p>
                        <p><em><strong>Request cost</strong> {llm_cost:.6f}s</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                return

        # Display structured JSON response with pastel colors

        bg_color = "#ffcdd2" if analysis["hazard_detected"] else "#c8e6c9"  # Light red / Light green
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 5px; background-color: {bg_color}; color: black;">
                <p><strong>Status:</strong> {'⚠️ HAZARD DETECTED' if analysis["hazard_detected"] else '✅ NO HAZARD DETECTED'}</p>
                <p><strong>Type:</strong> {analysis["hazard_type"]}</p>
                <p><strong>Reasoning:</strong> {analysis["reasoning"]}</p>
                <p><em><strong>LLM inference time </strong> {llm_inference_time:.1f}s</em></p>
                <p><em><strong>Latency (mosaic plot preprocessing + LLM inference time)</strong> {overall_inference_time:.1f}s</em></p>
                <p><em><strong>Request cost</strong> {llm_cost:.6f}s</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        # Fallback for any other errors with orange background
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 5px; background-color: #ffe5d0; color: black;">
                <p><strong>⚠️ Error Processing Response:</strong></p>
                <p><code>{str(analysis)}</code></p>
                <p><strong>Error:</strong> {str(e)}</p>
                <p><em><strong>LLM inference time </strong> {llm_inference_time:.1f}s</em></p>
                <p><em><strong>Latency (mosaic plot preprocessing + LLM inference time)</strong> {overall_inference_time:.1f}s</em></p>
                <p><em><strong>Request cost</strong> {llm_cost:.6f}s</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )


#message = {'Hazards detected': True, 'hazards_type': ['robbery'], 'reasoning': "Image_0: Armed individual appears in the field of view, presenting a threat to my position. - hazard detected: True - hazard type: ['robbery']\nImage_1: Unmasked and unarmed individuals are walking towards the car. - hazard detected: True - hazard type: []"}
#display_analysis(message)


def create_title_page(title: str = "") -> st.title:
    return st.title(title)


def create_start_analysis_button() -> st.button:
    return st.button(
        "▶️ Start Analysis",
        use_container_width=True,
        type="secondary"
    )


# ToDo: This should be a connector interface with different implementation, e.g.: video from path, video from cam, etc.
def video_connector(path: str) -> Tuple:
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    return video, total_frames, fps


def col_video_data_display(
        video_in: cv2.VideoCapture,
        frame_num: int,
        col_title: str = "**Video Frame**"
) -> None:

    total_frames = st.session_state.video_metadata['total_frames']
    current_frame = int(video_in.get(cv2.CAP_PROP_POS_FRAMES))
    video_in.set(cv2.CAP_PROP_POS_FRAMES, min(frame_num, total_frames-1))
    ret, frame = video_in.read()
    fps = video_in.get(cv2.CAP_PROP_FPS)

    if ret:
        st.markdown(col_title)
        frame_time = frame_num / fps
        st.image(
            process_frame(frame),
            caption=f"Frame {frame_num} ({format_time(frame_time)})"
        )

    video_in.set(cv2.CAP_PROP_POS_FRAMES, current_frame)


def col_mosaic_plot_visualization(
        mosaic_plots: List[np.ndarray],
        title: str = "**Images analyzed by LLM**"
) -> None:
    st.markdown(title)

    if len(mosaic_plots) == 0:
        st.write(f"**No Image selected**")
        return

    captions = [f"Image {i + 1}" for i in range(len(mosaic_plots))]

    cols = st.columns(len(mosaic_plots))

    for idx, col in enumerate(cols):
        with col:
            st.image(
                mosaic_plots[idx],
                caption=captions[idx],
                use_container_width=True  # Updated to suppress the warning
            )


def convert_minutes_seconds_to_frames(minutes: float, seconds: float) -> int:
    # Convert to MM:SS format
    metadata = st.session_state.video_metadata
    time_in_seconds = minutes * 60 + seconds
    fps = metadata['fps']
    frames = int(time_in_seconds * fps)

    st.sidebar.write(f"**Info**: time {time_in_seconds:.2f} seconds corresponds to frame number {frames}")

    return frames


def create_minute_seconds_button(title: str, key_minutes: str = "minutes", key_seconds: str = "seconds",
                                 expander: bool = False) \
        -> Tuple[float, float, int]:
    metadata = st.session_state.video_metadata

    # Using Streamlit's native expander for collapsible content
    with st.sidebar.expander(title, expanded=expander):
        # Layout with columns for minutes and seconds
        col1, col2 = st.columns(2)

        with col1:
            minutes = st.number_input("Minutes", min_value=0, max_value=metadata['total_frames'], value=0, format="%d",
                                      step=1, key=key_minutes)

        with col2:
            seconds = st.number_input("Seconds", min_value=0, max_value=metadata['total_frames'], value=0, format="%d",
                                      step=1, key=key_seconds)

        metadata = st.session_state.video_metadata
        time_in_seconds = minutes * 60 + seconds
        fps = metadata['fps']
        frames = int(time_in_seconds * fps)

        st.write(f"**Info**: time {time_in_seconds:.2f} seconds corresponds to frame number {frames}")

    return float(minutes), float(seconds), frames


def create_minute_seconds_buttons_and_get_frame(title: str, key_minutes: str = "minutes", key_seconds: str = "seconds",
                                                expander: bool = False) -> int:

    minutes, seconds, frames = create_minute_seconds_button(title, key_minutes, key_seconds, expander)
    return frames


def create_mosaic_plot_dimension_icon():
    with st.sidebar.expander(f"#### **Select mosaic plot dimension**"):

        # Layout with columns for minutes and seconds

        mosaic_plot_dim = st.number_input(
            "mosaic plot dim.",
            min_value=1,
            max_value=10,
            value=5,
            format="%d",
            step=1,
            key="dimension",
            help=f"Larger dimensions make analysis harder but reduce costs by fitting more frames into one "
                 f"image mosaic plot."
        )

        st.session_state.mosaic_plot_dim = mosaic_plot_dim

        return mosaic_plot_dim


def show_video():
    st.sidebar.video(st.session_state.video_file)


def create_llm_models_menu():
    model_names = []
    model_names.extend(LlmModelFactory.get_model_names())
    model_name = st.selectbox("Select Model", options=model_names, index=0)
    model_name_cost = LlmModelFactory[model_name].get_llm_model().get_euro_cost_per_million_tokens()
    model_name_cost["provider"] = LlmModelFactory[model_name].get_llm_model().get_provider_name()

    json_str = json.dumps(model_name_cost, indent=4)

    with st.sidebar.expander(f"**{model_name} metadata info**"):
        st.write(json_str)

    return model_name


def create_batch_size_and_frame_to_skip_frequency_buttons() -> Tuple[int | str, int]:
    with st.sidebar.expander(f"#### **Batch size and frames to skip freq.**"):

        # Layout with columns for minutes and seconds
        col1, col2 = st.columns(2)

        with col1:
            batch_size_options = ["Full"] + [str(i) for i in range(1, 17)]  # "Full" and numbers from 1 to 16
            batch_size = st.selectbox(
                "Batch size",
                options=batch_size_options,
                index=0,
                format_func=lambda x: "Full" if x == "Full" else f"{int(x)}",
                key="num_batches",
                help=f"Select the last 'Batch size' mosaic plots"
            )

        with col2:
            frames_to_skip_freq = st.number_input(
                "Skipped frames", min_value=0, max_value=120, value=6, format="%d", step=1, key="Frame skip frequency",
                help="Select frames at intervals; a skip frequency of 2 yields frames 1, 4, 7, 10."
            )

        fps = st.session_state.video_metadata["fps"]
        st.write(
            f"**Info**: Skip {frames_to_skip_freq} frames is equivalent to a time gap of "
            f"{frames_to_skip_freq / fps:.2f} seconds.")

        st.session_state.frames_to_skip_freq = frames_to_skip_freq

        return batch_size, frames_to_skip_freq


def visualize_current_frame_inference_frame_and_mosaic_plots_for_inference(
        llm_outputs: List[LlmOutput],
        current_frame_idx: int,
        old_inference_frame_idx: int,
        video: cv2.VideoCapture,
        current_inference_frame_idx: int,
        mosaic_chunks: List[np.ndarray],
        overall_inference_time: float,
        is_video_finished: bool
):

    def seconds_to_min_sec(seconds: float) -> str:
        minutes, whole_seconds = divmod(int(seconds), 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{minutes:02}:{whole_seconds:02}:{milliseconds:03}"


    #def seconds_to_min_sec(seconds: int) -> str:
    #    minutes, secs = divmod(seconds, 60)
    #    return f"{minutes:02}:{secs:02}"

    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if not is_video_finished:
                col_video_data_display(
                    video_in=video,
                    frame_num=current_frame_idx,
                    col_title="**Current Video Frame**"
                )

        with col2:
            col_video_data_display(
                video_in=video,
                frame_num=current_inference_frame_idx,
                col_title="**Inference Video Frame**"
            )

        with col3:
            fps = st.session_state.video_metadata["fps"]
            previous_inf_time = seconds_to_min_sec(old_inference_frame_idx/fps)
            current_inf_time = seconds_to_min_sec(current_inference_frame_idx/fps)
            col_mosaic_plot_visualization(
                mosaic_plots=mosaic_chunks,
                title=f"**Images analyzed by LLM in the period:** {previous_inf_time} - {current_inf_time}"
            )

        st.write("#### Inference results:")

        for llm_output in llm_outputs:
            display_analysis(
                analysis=llm_output.response,
                llm_inference_time=llm_output.inference_time,
                overall_inference_time=overall_inference_time,
                llm_cost=llm_output.cost,

            ),
        st.divider()


def get_frames_from_pool(idx_1: int, idx_2: int, frame_pool: List[np.ndarray]) -> List[np.ndarray]:
    return frame_pool[idx_1:idx_2]


def create_thread_label_buttons() -> Tuple[int, int] | None:
    st.sidebar.markdown("<br>", unsafe_allow_html=False)
    st.session_state.test_evaluation = st.sidebar.checkbox("Enable Evaluation Test")

    def seconds_to_min_sec(seconds: int) -> str:
        minutes, secs = divmod(seconds, 60)
        return f"{minutes:02}:{secs:02}"

    if st.session_state.test_evaluation:

        st.sidebar.markdown(
            """
            <div style="font-size:20px;">
                <strong>EVALUATION TEST</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state.number_evaluations = st.sidebar.number_input("Num. evaluations", min_value=1, max_value=1000,
                                                                      value=1, format="%d", step=1)

        initial_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Select initial threat time**",
            key_minutes="initial_thread_minutes",
            key_seconds="initial_thread_seconds",
            expander=True,
        )

        final_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Select final threat time**",
            key_minutes="final_thread_minutes",
            key_seconds="final_thread_seconds",
            expander=True
        )

        # Initialize list to store time pairs if not already initialized
        if 'initial_final_threat_frames_labels' not in st.session_state:
            st.session_state.initial_final_threat_frames_labels = []

        # Button to add current initial and final times to the list
        if st.sidebar.button(
            "Add threat range",
            help="Add ranges where a threat is identified through the 'Select initial/final threat time' widget "
                 "inputs above; this serves as a simple data labeling tool."
        ):
            if initial_thread_frame >= final_thread_frame:
                st.sidebar.error("Initial thread time must be less than final thread time.")
            else:
                st.session_state.initial_final_threat_frames_labels.append(
                    (initial_thread_frame, final_thread_frame)
                )

        # Display the list of tracked times
        st.sidebar.markdown("### Added labels:")
        fps = st.session_state.video_metadata["fps"]
        for idx, (start_frame, end_frame) in enumerate(st.session_state.initial_final_threat_frames_labels):
            start_time = seconds_to_min_sec(int(start_frame / fps))
            end_time = seconds_to_min_sec(int(end_frame / fps))

            # Create a horizontal layout with time info and remove button
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(
                    f"{idx + 1}. Start: {start_time} ({start_frame} frames), "
                    f"End: {end_time} ({end_frame} frames)"
                )
            with col2:
                # When remove button is pressed, modify the session state list
                if st.sidebar.button(f"Remove {idx + 1}", key=f"remove_{idx}"):
                    # Remove the item from the list
                    st.session_state.initial_final_threat_frames_labels.pop(idx)
                    # Trigger a state change (no need for rerun)
                    st.session_state.updated = True

        # Return the most recently added times or None if none added
        if st.session_state.initial_final_threat_frames_labels:
            return st.session_state.initial_final_threat_frames_labels[-1]
        else:
            return None
    else:
        return None


# ToDo: Remove this as this data should be saved at the end of the evaluation loop.
def create_test_label_button_save(title: str, path_to_tests: str = "test_evaluation_data"):
    def create_test_labels_data_and_save():

        if st.session_state.test_evaluation:
            current_date_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = st.session_state.video_file.name
            save_folder = os.path.join(path_to_tests, video_name, current_date_identifier)
            os.makedirs(save_folder, exist_ok=True)

            st.session_state.current_test_path = save_folder

            frame_info_path = os.path.join(save_folder, "labels.txt")
            labels = {
                "initial_thread_frame": st.session_state.initial_thread_frame,
                "final_thread_frame": st.session_state.final_thread_frame
            }

            with open(frame_info_path, "w") as f:
                for label, value in labels.items():
                    f.write(f"{label}: {value}\n")

            st.sidebar.success(f"Data saved to {save_folder}")

    if st.session_state.test_evaluation:
        if st.sidebar.button(title, key="save_button"):
            create_test_labels_data_and_save()


