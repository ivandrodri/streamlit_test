import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt, gridspec
from pydantic import BaseModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.backend.factories.llm_factory import LlmOutput


def get_filtered_directories_from_path_and_metadata(
        path: Path,
        frames_skip_freq: Optional[int] = None,
        grid_dim: Optional[int] = None,
        video_name: Optional[str] = None
) -> (List[Path], Dict):
    """
    Filters directories within the given path based on specified filters:
    frames_skip_freq, grid_dim, and name.

    :param path: The directory path where the search will occur.
    :param frames_skip_freq: The frames_skip_freq value to filter by.
    :param grid_dim: The grid_dim value to filter by.
    :param video_name: The base name of the video to filter by.
    :return: list of filtered directories (Path objects).
    """

    def count_unique_strings(strings_list: List[str]):
        counted_strings = Counter(strings_list)
        return list(counted_strings.items())

    dirs = [d for d in path.iterdir() if d.is_dir()]

    filtered_dirs = []
    video_names = []

    for dir_path in dirs:
        dir_name = dir_path.name  # Get the directory name as a string
        freq_extracted = int(dir_name.split('frames_skip_freq_')[1].split('_')[0])
        video_name_extracted = dir_name.split('_frames_skip_freq')[0]
        grid_dim_extracted = int(dir_name.split('_grid_dim_')[1].split('_')[0])

        if (frames_skip_freq is None or frames_skip_freq == freq_extracted) and \
                (grid_dim is None or grid_dim == grid_dim_extracted) and \
                (video_name is None or video_name == video_name_extracted):
            filtered_dirs.append(dir_path)
            video_names.append(video_name_extracted)

    metadata = {
        "video_name_and_freq": count_unique_strings(video_names)
    }

    return filtered_dirs, metadata


class CollectedDataForEvaluation(BaseModel):
    precisions: Dict[str, List[float]]
    recalls: Dict[str, List[float]]
    f1_scores: Dict[str, List[float]]
    supports: Dict[str, List[float]]
    accuracies: List[float]
    llm_latencies: List[List[float]]
    number_frames_analyzed: List[int] | None = None


def extract_data_from_folders(
        folder_paths: List[Path],
        labels_to_collect: Tuple[str, ...] = ('Non-Hazard', 'Hazard')
):
    accuracies = []
    llm_latencies = []
    number_frames_analyzed = []

    precisions = {label: [] for label in labels_to_collect}
    recalls = {label: [] for label in labels_to_collect}
    supports = {label: [] for label in labels_to_collect}
    f1_scores = {label: [] for label in labels_to_collect}

    for folder in folder_paths:
        accuracy_file = folder / "accuracy_and_mean_latencies.txt"
        classification_file = folder / "classification_report.csv"

        # Extract accuracy and llm latencies from the "call accuracy_and_mean_latencies.txt"
        if os.path.exists(accuracy_file):
            with open(accuracy_file, "r") as file:
                lines = file.readlines()
                accuracy = None
                llm_latencies_data = None
                for line in lines:
                    if "Accuracy" in line:
                        accuracy = float(line.split(":")[1].strip())
                    if "llm_latencies" in line:
                        llm_latencies_data = eval(line.split(":")[1].strip())
                    if "number_frames_analyzed" in line:
                        number_frames = eval(line.split(":")[1].strip())

                if accuracy is not None:
                    accuracies.append(accuracy)
                if llm_latencies_data is not None:
                    llm_latencies.append(llm_latencies_data)
                if number_frames is not None:
                    number_frames_analyzed.append(number_frames)

        # Extract precision, recall, and support from the "classification_report.csv"
        if os.path.exists(classification_file):
            df = pd.read_csv(classification_file)
            for index, row in df.iterrows():
                if row['Unnamed: 0'] in labels_to_collect:
                    precisions[row['Unnamed: 0']].append(row["precision"])
                    f1_scores[row['Unnamed: 0']].append(row["f1-score"])
                    supports[row['Unnamed: 0']].append(row["support"])
                    recalls[row['Unnamed: 0']].append(row["recall"])

    return CollectedDataForEvaluation(
        precisions=precisions,
        recalls=recalls,
        f1_scores=f1_scores,
        supports=supports,
        accuracies=accuracies,
        llm_latencies=llm_latencies,
        number_frames_analyzed=number_frames_analyzed
    )


def report_plot(
        collected_data_metadata: Dict[str, int],
        collected_data: CollectedDataForEvaluation,
        save: bool = False,
        path_to_save: Path | None = None
):
    """
    Plot a bar chart of video frequencies and a scatter plot of LLM latencies, with summary text and two tables.
    :param path_to_save:
    :param save:
    :param collected_data_metadata:
    :param collected_data:
    :return:
    """

    def create_inset_message_first_plot(ax):
        # Add an inset to the frequency plot
        inset_ax = ax.inset_axes([0.7, 0.7, 0.25, 0.25])  # Position inset (x, y, width, height)
        inset_message = f"Total Videos Analyzed: {total_videos} \n"
        tot_num_frames = sum(collected_data.number_frames_analyzed)
        inset_message += f"Total frames analyzed: {tot_num_frames} \n"
        num_inference_calls = sum(len(sublist) for sublist in collected_data.llm_latencies)
        inset_message += f"Total LLM requests: {num_inference_calls} \n"
        inset_message += f"Num. frames x LLM requests: {(tot_num_frames / num_inference_calls):.2f} \n"
        inset_ax.text(0.7, 0.0, inset_message,
                      ha="center", va="center", fontsize=8, fontweight="bold", color="green")
        inset_ax.axis("off")

    def get_llm_latencies():
        return [latency for sublist in collected_data.llm_latencies for latency in sublist]

    def table_cosmetic(table, color="#F08080", column=0, fontsize=12):
        # Color customization for the first table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_fontsize(fontsize)
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D3D3D3')  # Light gray for header
            else:
                cell.set_fontsize(10)
                if j == column:  # Value column, different color for values
                    cell.set_facecolor(color)  # Light yellow for values

        # Create another table below (in the new row)
        ax4 = fig.add_subplot(gs[2, :])  # The new row, spanning both columns
        ax4.axis("off")  # Disable axis for this plot

    def create_table_from_dataframe(ax, table_df, table_title, fontsize=12):

        ax.axis("off")  # Disable axis

        # Add text
        ax.text(0.5, 1.2, table_title, ha="center", va="center", fontsize=fontsize, fontweight="bold")

        # Add the first table
        table = ax.table(cellText=table_df.values,
                          colLabels=table_df.columns,
                          cellLoc='center', loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.5)

        return table

    def get_mean_dict_and_keys(dict: Dict) -> Tuple[List[float], List[str]]:
        means = []
        keys = []
        for key, value in dict.items():
            means.append(np.mean(value))
            keys.append(key)
        return means, keys

    video_names = [elem[0] for elem in collected_data_metadata]
    frequencies = [int(elem[1]) for elem in collected_data_metadata]
    total_videos = sum(frequencies)

    # Create a 3-row, 2-column grid (increase to 3 rows for an additional table)
    fig = plt.figure(figsize=(14, 10))  # Increased height for the extra row
    gs = gridspec.GridSpec(3, 2, height_ratios=[7, 5, 5], width_ratios=[1, 1], hspace=3.5)
    fig.suptitle("EVALUATION REPORT", fontsize=16, fontweight='bold')

    # Frequency plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])  # Span both columns
    ax1.bar(video_names, frequencies, color='skyblue')
    ax1.set_xlabel("Video Names")
    ax1.set_ylabel("Number of videos")
    ax1.set_title("Frequency of Videos Analyzed")
    ax1.tick_params(axis='x', rotation=45)

    ax1b = fig.add_subplot(gs[0, 1])
    tot_num_frames = sum(collected_data.number_frames_analyzed)
    num_inference_calls = sum(len(sublist) for sublist in collected_data.llm_latencies)
    videos_metadata_table = pd.DataFrame({
        'Total videos': [total_videos],
        'Total frames': [tot_num_frames],
        'Total LLM requests': [num_inference_calls],
        'frames/requests': [round((tot_num_frames / num_inference_calls), 2)],
    })

    table_video_metadata = create_table_from_dataframe(
        ax=ax1b,
        table_df=videos_metadata_table,
        table_title="  ",
        fontsize=8
    )
    table_cosmetic(table_video_metadata, fontsize=8, color="white")



    # LLM Latencies plot (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left plot
    llm_latencies = get_llm_latencies()
    x_indices = range(1, len(llm_latencies) + 1)  # Use indices for x-axis

    ax2.scatter(x_indices, llm_latencies, color='lightcoral', label='LLM Latencies')
    ax2.axhline(y=float(np.mean(llm_latencies)), color='black', linestyle='--', label='Mean Latency')
    ax2.set_xlabel("Inference Call Index")
    ax2.set_ylabel("LLM Latency (seconds)")
    ax2.set_title("LLM Latencies and Mean")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])

    precisions, labels = get_mean_dict_and_keys(collected_data.precisions)
    recalls, _ = get_mean_dict_and_keys(collected_data.recalls)
    f1_scores, _ = get_mean_dict_and_keys(collected_data.f1_scores)

    precision_recall_table = pd.DataFrame({
        'Label': labels,
        'Mean Precision': [f"{precision:.2f}" for precision in precisions],
        'Mean Recall': [f"{recall:.2f}" for recall in recalls],
        'Mean f1_score': [f"{f1_score:.2f}" for f1_score in f1_scores],
    })

    table_accuracy = create_table_from_dataframe(ax=ax3, table_df=precision_recall_table,
                                                 table_title="Performance metrics \n \n",
                                                 fontsize=10)
    table_cosmetic(table_accuracy, fontsize=10)

    # Create another table below (in the new row)
    ax4 = fig.add_subplot(gs[2, :])  # The new row, spanning both columns

    summary_df = pd.DataFrame({
        'Metric': ['Mean Accuracy', 'Mean LLM Latency'],
        'Value': [
            f"{np.mean(collected_data.accuracies):.2f}",
            f"{np.mean(get_llm_latencies()):.2f}"
        ]
    })

    table_precision = create_table_from_dataframe(ax=ax4, table_df=summary_df, table_title="KPI metrics \n \n")
    table_cosmetic(table_precision)

    if save:
        if path_to_save.exists():
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = path_to_save / f"report_{current_time}.pdf"
            fig.savefig(file_path, format='pdf')
            logging.info(f"Report saved to {file_path}")

    plt.tight_layout()
    plt.show()


def data_collected_report(
        model_name: str,
        collected_data: CollectedDataForEvaluation,
        collected_data_metadata: Dict[str, int],
        save: bool = False,
        path_to_save: Path | None = None
):
    # Header
    report = f"### Data Collection Report for {model_name} ###\n\n"

    # Section: Files Analyzed
    report += "#### Files Analyzed: Video Names and Frequency ####\n"

    # Initialize video analysis summary
    num_videos = 0
    video_list = []

    for video_name, frequency in collected_data_metadata:
        video_list.append(f"{video_name}: {frequency}")
        num_videos += frequency

    # Add the list of videos analyzed
    report += "\n".join(video_list) + "\n\n"
    report += f"Total videos analyzed: {num_videos}\n"

    # Optional: Add Visualization (bar chart of frequencies)
    report_plot(
        collected_data_metadata=collected_data_metadata,
        collected_data=collected_data,
        save=save,
        path_to_save=path_to_save
    )


def save_and_display_classification_report_to_csv(
        true_labels: List,
        predicted_labels: List,
        llm_latencies: List,
        overall_latencies: List,
        selected_model_name: str,
        path_to_tests: str = "test_evaluation_data",
        number_frames_analyzed: int | None = None,
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
        file_test_name = f"{video_name}_{frames_to_skip_prefix}_{mosaic_plot_dim_prefix}_{current_date_identifier}"

        model_name = selected_model_name
        save_folder = os.path.join(path_to_tests, model_name, file_test_name)
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
            f.write(f"llm_latencies: {llm_latencies}\n")
            f.write(f"number_frames_analyzed: {number_frames_analyzed}\n")
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

            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Hazard", "Hazard"],
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
