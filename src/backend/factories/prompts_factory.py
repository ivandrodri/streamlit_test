from enum import StrEnum

from src.backend.prompts import HAZARD_DETECTION_PROMPT_SINGLE_FRAME, HAZARD_DETECTION_MULTIPLE_FRAMES_SINGLE_BATCH, \
    HAZARD_DETECTION_MULTIPLE_FRAMES_SINGLE_BATCH_CoT


class PromptsFactory(StrEnum):
    hazard_detection_prompt = "hazard_detection_prompt"
    hazard_detection_multiple_frames_single_batch = "hazard_detection_multiple_frames_single_batch"
    hazard_detection_multiple_frames_single_batch_cot = "hazard_detection_multiple_frames_single_batch_cot"

    def get_prompt(self):
        match self:
            case self.hazard_detection_prompt:
                return HAZARD_DETECTION_PROMPT_SINGLE_FRAME
            case self.hazard_detection_multiple_frames_single_batch:
                return HAZARD_DETECTION_MULTIPLE_FRAMES_SINGLE_BATCH
            case self.hazard_detection_multiple_frames_single_batch_cot:
                return HAZARD_DETECTION_MULTIPLE_FRAMES_SINGLE_BATCH_CoT
