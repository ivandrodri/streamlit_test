import json
import logging
import os
from datetime import datetime
from typing import Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from backend.factories.llm_factory import LlmModelFactory, LlmOutput
from backend.utils import process_frame, format_time
import seaborn as sns

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


def create_minute_seconds_button(title: str, key_minutes: str = "minutes", key_seconds: str = "seconds") \
        -> Tuple[float, float]:
    st.sidebar.write(title)

    # Layout with columns for minutes and seconds
    col1, col2 = st.sidebar.columns(2)
    metadata = st.session_state.video_metadata

    with col1:
        minutes = st.number_input("Minutes", min_value=0, max_value=metadata['total_frames'], value=0, format="%d",
                                  step=1, key=key_minutes)

    with col2:
        seconds = st.number_input("Seconds", min_value=0, max_value=metadata['total_frames'], value=0, format="%d",
                                  step=1, key=key_seconds)

    return float(minutes), float(seconds)


def create_minute_seconds_buttons_and_get_frame(title: str, key_minutes: str = "minutes", key_seconds: str = "seconds") \
        -> int:

    minutes, seconds = create_minute_seconds_button(title, key_minutes, key_seconds)
    return convert_minutes_seconds_to_frames(minutes, seconds)


def create_mosaic_plot_dimension_icon():
    st.sidebar.write(f"#### **Choose mosaic plot dimension**")

    # Layout with columns for minutes and seconds

    mosaic_plot_dim = st.sidebar.number_input("mosaic plot dim.", min_value=1, max_value=10, value=5, format="%d",
                                              step=1, key="dimension")

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
    st.sidebar.markdown(f"<details><summary>{model_name} info </summary><pre>{json_str}</pre></details>",
                        unsafe_allow_html=True)
    return model_name


def create_batch_size_and_frame_to_skip_frequency_buttons() -> Tuple[int | str, int]:
    st.sidebar.write(f"#### **Batch size and frames to skip freq.**")

    # Layout with columns for minutes and seconds
    col1, col2 = st.sidebar.columns(2)

    with col1:
        batch_size_options = ["Full"] + [str(i) for i in range(1, 17)]  # "Full" and numbers from 1 to 16
        batch_size = st.selectbox(
            "Batch size",
            options=batch_size_options,
            index=0,
            format_func=lambda x: "Full" if x == "Full" else f"{int(x)}",
            key="num_batches"
        )

    with col2:
        frames_to_skip_freq = st.number_input("Frames to skip freq.", min_value=0, max_value=120, value=6, format="%d",
                                              step=1, key="frames to skip")

    fps = st.session_state.video_metadata["fps"]
    st.sidebar.write(
        f"**Info**: Skip {frames_to_skip_freq} frames is equivalent to a time gap of {frames_to_skip_freq / fps:.2f} "
        f"seconds.")

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

        st.write("#### Overall Analysis:")

        for llm_output in llm_outputs:
            display_analysis(
                analysis=llm_output.response,
                llm_inference_time=llm_output.inference_time,
                overall_inference_time=overall_inference_time,
                llm_cost=llm_output.cost,

            ),
        st.divider()


def get_true_label_prediction_and_tokens_per_second(
        llm_outputs: List[LlmOutput],
        thread_frame_labels: List[Tuple[int, int]],
        old_inference_frame_idx: int,
        current_frame_idx: int,
) -> Tuple[bool, bool] | None:

    if len(thread_frame_labels) > 0:

        def thread_detected_from_llm(response: Dict) -> bool:
            if response["hazard_detected"]:
                return True
            return False

        def is_real_hazard_in_time_window(old_frame: int, current_frame: int, thread_frames: List[Tuple[int, int]]) \
                -> bool:
            def is_element_within_range(element, range_tuple) -> bool:
                return range_tuple[0] <= element <= range_tuple[1]

            is_thread_in_current_window = False
            for thread_frame in thread_frames:
                if (is_element_within_range(old_frame, thread_frame) or
                        is_element_within_range(current_frame, thread_frame)):
                    is_thread_in_current_window = True
                    break

            return is_thread_in_current_window

        prediction = False
        for llm_output in llm_outputs:
            if thread_detected_from_llm(llm_output.response):
                prediction = True
                break

        label = is_real_hazard_in_time_window(old_inference_frame_idx, current_frame_idx, thread_frame_labels)
        return label, prediction
    else:
        logging.info(f"thread_frame_labels is empty. Please add the labels in the UI.")
        return None


def get_frames_from_pool(idx_1: int, idx_2: int, frame_pool: List[np.ndarray]) -> List[np.ndarray]:
    return frame_pool[idx_1:idx_2]


def create_thread_label_buttons() -> Tuple[int, int] | None:
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
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

        initial_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Choose initial threat time**",
            key_minutes="initial_thread_minutes",
            key_seconds="initial_thread_seconds"
        )

        final_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Choose final threat time**",
            key_minutes="final_thread_minutes",
            key_seconds="final_thread_seconds"
        )

        # Initialize list to store time pairs if not already initialized
        if 'initial_final_threat_frames_labels' not in st.session_state:
            st.session_state.initial_final_threat_frames_labels = []

        # Button to add current initial and final times to the list
        if st.sidebar.button("Add Time Pair"):
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

'''
def create_thread_label_buttons() -> Tuple[int, int] | None:
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
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

        initial_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Choose initial threat time**",
            key_minutes="initial_thread_minutes",
            key_seconds="initial_thread_seconds"
        )

        final_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Choose final threat time**",
            key_minutes="final_thread_minutes",
            key_seconds="final_thread_seconds"
        )

        # Initialize list to store time pairs if not already initialized
        if 'initial_final_threat_frames_labels' not in st.session_state:
            st.session_state.initial_final_threat_frames_labels = []

        # Button to add current initial and final times to the list
        if st.sidebar.button("Add Time Pair"):
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
            start_time = seconds_to_min_sec(int(start_frame/fps))
            end_time = seconds_to_min_sec(int(end_frame/fps))

            st.sidebar.write(
                f"{idx + 1}. Start: {start_time} ({start_frame} frames), "
                f"End: {end_time} ({end_frame} frames)"
            )

        # Return the most recently added times or None if none added
        if st.session_state.initial_final_threat_frames_labels:
            return st.session_state.initial_final_threat_frames_labels[-1]
        else:
            return None
    else:
        return None
'''

'''
def create_thread_label_buttons() -> Tuple[int, int] | None:
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.session_state.test_evaluation = st.sidebar.checkbox("Enable Evaluation Test")

    if st.session_state.test_evaluation:

        st.sidebar.markdown(
            """
            <div style="font-size:20px;">
                <strong>EVALUATION TEST</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state.initial_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Choose initial thread time**",
            key_minutes="initial_thread_minutes",
            key_seconds="initial_thread_seconds"
        )

        st.session_state.final_thread_frame = create_minute_seconds_buttons_and_get_frame(
            title=f"#### **Choose final thread time**",
            key_minutes="final_thread_minutes",
            key_seconds="final_thread_seconds"
        )

        if st.session_state.initial_thread_frame >= st.session_state.final_thread_frame:
            st.sidebar.error("Initial thread time must be less than final thread time.")
            return None



        return st.session_state.initial_thread_frame, st.session_state.final_thread_frame
    else:
        return None
'''

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


def save_and_display_classification_report_to_csv(
        true_labels: List,
        predicted_labels: List,
        llm_latencies: List,
        overall_latencies: List,
        path_to_tests: str = "test_evaluation_data"
):

    def nice_visualization(key: str, quantity: float):
        st.markdown(
            f"""
                       <div style="font-size:24px; font-weight:bold; color:green; margin-bottom:20px;">
                           {key}: {quantity:.4f} 
                       </div>
                       """,
            unsafe_allow_html=True
        )

    if st.session_state.test_evaluation:
        video_name = st.session_state.video_file.name

        frames_to_skip_prefix = f"frames_skip_freq_{st.session_state.frames_to_skip_freq}"
        mosaic_plot_dim_prefix = f"grid_dim_{st.session_state.mosaic_plot_dim}"
        current_date_identifier = f"date_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_test_name = f"{frames_to_skip_prefix}_{mosaic_plot_dim_prefix}_{current_date_identifier}"

        save_folder = os.path.join(path_to_tests, video_name, file_test_name)
        os.makedirs(save_folder, exist_ok=True)

        # Generate Classification Report
        class_report = classification_report(
            y_true=true_labels,
            y_pred=predicted_labels,
            target_names=["Non-Hazard", "Hazard"],
            labels=[0, 1],  # Specify labels explicitly
            output_dict=True
        )

        # Calculate Accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Convert to DataFrame
        df_report = pd.DataFrame(class_report).transpose()

        # Drop the 'accuracy' row
        df_report = df_report.drop(index='accuracy')

        # Reorder the columns (if needed)
        df_report = df_report[['precision', 'recall', 'f1-score', 'support']]

        # Save the report to a CSV file
        csv_file_path = os.path.join(save_folder, "classification_report.csv")
        df_report.to_csv(csv_file_path, index=True)
        logging.info(f"CSV report saved to: {csv_file_path}")

        # Save the accuracy to a separate text file
        accuracy_file_path = os.path.join(save_folder, "accuracy_and_mean_latencies.txt")
        with open(accuracy_file_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f} \n")
            f.write(f"llm_mean_latency: {np.mean(llm_latencies):.4f} \n")
            f.write(f"overall_mean_latency: {np.mean(overall_latencies):.4f} \n")
        logging.info(f"Accuracy and latencies saved to: {accuracy_file_path}")

        # Generate Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # Create a Streamlit columns layout to display both the table and confusion matrix
        col1, col2 = st.columns([2.0, 1.5])  # Adjust the ratio as needed

        # Display the classification report on the left column
        with col1:
            st.write("#### Classification Report")
            st.dataframe(df_report, width=700, height=200)

            nice_visualization("Accuracy", accuracy)
            nice_visualization("LLM mean latency", float(np.mean(llm_latencies)))
            nice_visualization("Overall mean latency", float(np.mean(overall_latencies)))

        with col2:
            font_size = 15
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 6))  # Set the figure size for the confusion matrix

            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Hazard", "Non-Hazard"],
                        yticklabels=[ "Non-Hazard", "Hazard"], ax=ax, cbar=False,
                        annot_kws={"size": 5, "weight": 'bold'})

            ax.set_xlabel('Predicted', fontsize=font_size, weight='bold')
            ax.set_ylabel('True', fontsize=font_size, weight='bold')

            ax.set_aspect('equal')

            plt.xticks(rotation=45, ha='right', fontsize=font_size)
            plt.yticks(rotation=45, ha='right', fontsize=font_size)

            for text in ax.texts:
                text.set_fontsize(font_size)
                text.set_fontweight('bold')
                text.set_color('black')

            st.pyplot(fig)

        return csv_file_path, accuracy_file_path
