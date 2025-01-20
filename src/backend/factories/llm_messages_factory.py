import base64
import io
from enum import StrEnum
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image


class LlmMessageFactory(StrEnum):
    single_frame_message_openai = "single_frame_message_openai"
    single_frame_message_fireworks = "single_frame_message_fireworks"
    multi_frame_message_openai = "multi_frame_message_openai"

    @staticmethod
    def _encode_image(frame):
        """Convert CV2 frame to base64 string"""
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_message(self, prompt: str, frames: np.ndarray | List[np.ndarray]) -> List[Dict]:
        match self:

            case self.single_frame_message_openai:
                if not isinstance(frames, np.ndarray):
                    raise ValueError(f"For a single_frame_message, frames should be np.ndarray "
                                     f"but you provided {type(frames)}")

                base64_frame = self._encode_image(frames)
                return [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frame}"
                            }
                        }
                    ],
                    "stream": True,
                    "stream_options": {
                        "include_usage": True
                    }
                }]

            case self.single_frame_message_fireworks:
                if not isinstance(frames, np.ndarray):
                    raise ValueError(f"For a single_frame_message, frames should be np.ndarray "
                                     f"but you provided {type(frames)}")

                base64_frame = self._encode_image(frames)
                return [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frame}"
                            }
                        }
                    ],
                }]

            case self.multi_frame_message_openai:
                if not isinstance(frames, List):
                    raise ValueError(f"For a multi_frame_message, frames must be a List[np.ndarray]")

                base64_frame_batch = [self._encode_image(frame) for frame in frames]
                return [{
                    "role": "user",
                    "content": [
                                   {"type": "text", "text": prompt}
                               ] + [
                                   {
                                       "type": "image_url",
                                       "image_url": {
                                           "url": f"data:image/jpeg;base64,{frame}"
                                       }
                                   } for frame in base64_frame_batch
                               ],
                    "stream": True,
                    "stream_options": {
                        "include_usage": True
                    }
                }]
