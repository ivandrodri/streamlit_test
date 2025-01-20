import base64
import io
import logging
import os
from typing import List

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from joblib import Parallel, delayed

from backend.factories.llm_factory import LlmOutput, LlmModelFactory, LLMInput
from backend.factories.llm_messages_factory import LlmMessageFactory

load_dotenv(dotenv_path="../../.env")


logging.basicConfig(level=logging.INFO)

NUM_CORES = min(4, os.cpu_count())

def process_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def format_time(seconds):
    return f"{int(seconds/60):02d}:{int(seconds%60):02d}:{int((seconds % 1) * 1000):03d}"


def encode_image_to_base64(frame):
    """Convert CV2 frame to base64 string"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64


def single_mosaic_plot_generation(
        frames: List[np.ndarray],
        idx_mosaic_plot: int,
        n_cols: int,
        n_rows: int,
        frames_to_skip_frequency: int,
        start_index_subplot: int,
        save_image: bool,
        output_folder: str | None,
        idx: int | None,
):

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()  # Flatten the array of axes for easy indexing

    number_images_previous_mosaic = n_cols*n_rows
    if frames_to_skip_frequency != 0:
        number_images_previous_mosaic *= (frames_to_skip_frequency+1)

    subplot_frame = start_index_subplot + idx_mosaic_plot*number_images_previous_mosaic

    for sub_idx, frame in enumerate(frames):
        axes[sub_idx].imshow(frame)
        axes[sub_idx].axis('off')  # Hide axis
        if frames_to_skip_frequency == 0:
            subplot_frame += 1
        else:
            subplot_frame += (frames_to_skip_frequency + 1)
        axes[sub_idx].text(0.5, 0.95, str(subplot_frame), ha='center', va='center',
                           transform=axes[sub_idx].transAxes, fontsize=12, color='white',
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    # Hide any unused subplots
    for sub_idx in range(len(frames), len(axes)):
        axes[sub_idx].axis('off')

    plt.tight_layout()

    # Render figure to buffer
    fig.canvas.draw()
    rgba_buffer = fig.canvas.renderer.buffer_rgba()
    mosaic_array = np.frombuffer(rgba_buffer, dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    mosaic_array = mosaic_array.reshape((height, width, 4))
    mosaic_array = mosaic_array[:, :, :3]

    if save_image:
        image_name = f"mosaic_{idx_mosaic_plot}.png" if idx is None else f"mosaic_{idx_mosaic_plot}_{idx}.png"
        output_file_path = os.path.join(output_folder, image_name)
        plt.savefig(output_file_path)
        logging.info(f"Mosaic figure {image_name} saved to {output_file_path}")

    plt.close(fig)
    return mosaic_array


def create_mosaic_from_video_frames(
        frames: List[np.ndarray],
        output_folder: str | None = None,
        n_cols: int = 5,
        n_rows: int = 5,
        idx: int | None = None,
        save_image: bool = False,
        start_index_subplot: int = 0,
        batch_size: int | None = None,
        frames_to_skip_frequency: int = 0
) -> List[np.ndarray]:
    """
    Create batched of mosaic plots of n_rows X n_cols from frames.
    """

    if len(frames) == 0:
        logging.info(f"No frames provided for mosaic plot LLM preprocessing")
        return frames

    if len(frames) == 1:
        frames_to_skip_frequency = 0

    if save_image and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_to_be_preprocess = frames if frames_to_skip_frequency == 0 else \
        [frame for i, frame in enumerate(frames) if i % (frames_to_skip_frequency + 1) == 0]

    num_mosaic_plots = int(np.ceil(len(frames_to_be_preprocess) / (n_rows * n_cols)))

    if batch_size is not None:
        num_mosaic_plots = min(batch_size, num_mosaic_plots)

    logging.info(f"Total frames to be process for mosaic plots creation: {len(frames_to_be_preprocess)}")

    batch_mosaic_plots = []
    for idx_mosaic_plot in range(num_mosaic_plots):

        frames_mosaic_plot = frames_to_be_preprocess[idx_mosaic_plot*(n_cols*n_rows):(idx_mosaic_plot+1)*(n_cols*n_rows)]

        mosaic_array = single_mosaic_plot_generation(
                frames=frames_mosaic_plot,
                idx_mosaic_plot=idx_mosaic_plot,
                n_cols=n_cols,
                n_rows=n_rows,
                frames_to_skip_frequency=frames_to_skip_frequency,
                start_index_subplot=start_index_subplot,
                save_image=save_image,
                output_folder=output_folder,
                idx=idx,
        )

        batch_mosaic_plots.append(mosaic_array)

    return batch_mosaic_plots


def single_thread_mosaic_plot_creation(
        frames: List[np.ndarray],
        idx_mosaic_plot: int,
        n_cols: int,
        n_rows: int,
        frames_to_skip_frequency: int,
        start_index_subplot: int,
        save_image: bool,
        output_folder: str | None,
        idx: int | None,
):
    frames_mosaic_plot = frames[
                         idx_mosaic_plot * (n_cols * n_rows):(idx_mosaic_plot + 1) * (n_cols * n_rows)]

    mosaic_array = single_mosaic_plot_generation(
        frames=frames_mosaic_plot,
        idx_mosaic_plot=idx_mosaic_plot,
        n_cols=n_cols,
        n_rows=n_rows,
        frames_to_skip_frequency=frames_to_skip_frequency,
        start_index_subplot=start_index_subplot,
        save_image=save_image,
        output_folder=output_folder,
        idx=idx,
    )

    return mosaic_array


def create_mosaic_from_video_frames_multi_thread(
        frames: List[np.ndarray],
        output_folder: str | None = None,
        n_cols: int = 5,
        n_rows: int = 5,
        idx: int | None = None,
        save_image: bool = False,
        start_index_subplot: int = 0,
        batch_size: int | None = None,
        frames_to_skip_frequency: int = 0,
        max_threads: int = NUM_CORES
) -> List[np.ndarray]:
    """
    Create batched of mosaic plots of n_rows X n_cols from frames.
    """

    if len(frames) == 0:
        logging.info(f"No frames provided for mosaic plot LLM preprocessing")
        return frames

    if len(frames) == 1:
        frames_to_skip_frequency = 0

    if save_image and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_to_be_preprocess = frames if frames_to_skip_frequency == 0 else \
        [frame for i, frame in enumerate(frames) if i % (frames_to_skip_frequency + 1) == 0]

    num_mosaic_plots = int(np.ceil(len(frames_to_be_preprocess) / (n_rows * n_cols)))

    idx_selected_mosaic_plots = max(0, num_mosaic_plots - batch_size) if batch_size is not None else 0

    logging.info(f"Total frames to be process for mosaic plots creation: {len(frames_to_be_preprocess)}")

    batch_mosaic_plots = Parallel(n_jobs=max_threads)(
        delayed(single_thread_mosaic_plot_creation)(
            frames_to_be_preprocess,
            idx_mosaic_plot,
            n_cols,
            n_rows,
            frames_to_skip_frequency,
            start_index_subplot,
            save_image,
            output_folder,
            idx
        )

        for idx_mosaic_plot in range(idx_selected_mosaic_plots, num_mosaic_plots)
    )

    return batch_mosaic_plots


def create_images_pool_from_video(path: str) -> List[np.ndarray]:
    video = cv2.VideoCapture(path)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()

    return frames


def _aggregate_llm_outputs_for_fake_batch(llm_outputs: List[LlmOutput]) -> LlmOutput:

    def index_hazards_detected(hazards: List):
        for i, value in enumerate(hazards):
            if value is True:
                return i
        return None

    hazards_detected = []
    hazards_type = []
    reasoning = []
    cost = 0.0
    inference_time = 0.0
    for id_image, llm_output in enumerate(llm_outputs):
        hazards_detected.append(llm_output.response.get("hazard_detected", None))
        hazards_type.append(llm_output.response.get("hazard_type", None))
        reasoning.append(f"Image_{id_image}: {llm_output.response.get('reasoning')} - "
                         f"hazard detected: {str(hazards_detected[-1])} - "
                         f"hazard type: {str(hazards_type[-1])}")
        cost += llm_output.cost
        inference_time += llm_output.inference_time

    reasoning = "<br> ".join(reasoning)
    reasoning = "<br> " + reasoning
    index_hazard = index_hazards_detected(hazards_detected)

    hazard_detected = False if index_hazard is None else True
    hazard_type = "none" if index_hazard is None else hazards_type[index_hazard]
    response = {"hazard_detected": hazard_detected, "hazard_type": hazard_type, "reasoning": reasoning}

    return LlmOutput(
        response=response,
        inference_time=inference_time,
        cost=cost
    )


def get_llm_inference(
        model_name: str,
        frames: np.ndarray | List[np.ndarray],
        prompt: str,
        batch_mode: bool = False,
        min_response_time_fake_model: float = 0.0,
        max_response_time_fake_model: float = 0.0,
        max_num_tokens: float = 100
) -> List[LlmOutput]:

    if len(frames) == 0:
        return [LlmOutput(
            response="No frame was provided, llm request skipped",
            inference_time=0.0,
            cost=0.0
        )]

    llm_responses = []
    fake_batch_request = False

    llm_model = LlmModelFactory[model_name].get_llm_model()
    model_provider = llm_model.get_provider_name()

    if not batch_mode:

        if not isinstance(frames, np.ndarray):
            raise ValueError("Frames must be a numpy array in batch_mode=False .")

        # ToDo: Convert OPEN_AI and FIREWORKS to enums. This should be done in FactoryModels as they are uses
        #  in get_provider methods
        if "OPEN_AI" in model_provider:
            messages = LlmMessageFactory.single_frame_message_openai.get_message(prompt=prompt, frames=frames)
        elif "FIREWORKS" in llm_model.get_provider_name():
            messages = LlmMessageFactory.single_frame_message_fireworks.get_message(prompt=prompt, frames=frames)
        elif "fake_provider" in llm_model.get_provider_name():
            # We could use a fake prompt fot the fake model but fireworks one is used here for debugging.
            messages = LlmMessageFactory.single_frame_message_fireworks.get_message(prompt=prompt, frames=frames)
            #messages = LlmMessageFactory.single_frame_message_openai.get_message(prompt=prompt, frames=frames)
        else:
            raise ValueError(f"For batch_mode=False only OPEN_AI and FIREWORKS prompts are supported")
    else:
        if not isinstance(frames, List):
            raise ValueError("Frames must be a List of numpy arrays in batch_mode=True .")

        if llm_model.admit_batch_mode():
            messages = LlmMessageFactory.multi_frame_message_openai.get_message(prompt=prompt, frames=frames)
        # Fake batch to be used with multiple LLM calls
        else:
            fake_batch_request = True
            messages = None

    responses = []
    llm_cost = 0.0

    if fake_batch_request:

        llm_with_context_outputs = []
        for idx, frame in enumerate(frames):

            # ToDo: Try with all the previous responses and not only the last one.
            if idx == 0:
                prompt_with_context = f"Frame {idx + 1}:\n{prompt}"
            else:
                prompt_with_context = f"""
                    Previous analysis: {responses[-1]}

                    Now analyze the next frame (Frame {idx + 1}):
                    {prompt}
                    """

            message = LlmMessageFactory.single_frame_message_fireworks.get_message(
                prompt=prompt_with_context,
                frames=frame
            )

            model_input = LLMInput(
                model_name=model_name,
                message=message,
                max_tokens=max_num_tokens,
                fake_inference_time_range=(min_response_time_fake_model, max_response_time_fake_model)
            )

            llm_response = llm_model.get_response(model_input)
            responses.append(llm_response.response)

            llm_cost += llm_response.cost
            llm_with_context_outputs.append(llm_response)

            # Here we should aggregate the llm_responses

        llm_response = _aggregate_llm_outputs_for_fake_batch(llm_with_context_outputs)
        llm_responses.append(llm_response)

    else:

        model_input = LLMInput(
            model_name=model_name,
            message=messages,
            max_tokens=max_num_tokens,
            fake_inference_time_range=(min_response_time_fake_model, max_response_time_fake_model)
        )

        llm_response = llm_model.get_response(model_input)
        llm_cost += llm_response.cost

        llm_responses.append(llm_response)

    return llm_responses


'''
PATH = "/home/ivan/Documents/GIT_PROJECTS/Guardian_angel_BMW_innovation/GuardianAngel/data/XOOB4321_theft.MP4"
NUM_CORES = 8

frames = create_images_pool_from_video(path=PATH)

start_time = time.time()

batch_mosaic_plots = create_mosaic_from_video_frames_multi_thread(
    frames=frames[0:600],
    max_threads=NUM_CORES,
)

end_time = time.time()

print(f"Time {(end_time - start_time):.2f} seconds")
print(len(batch_mosaic_plots))

'''

