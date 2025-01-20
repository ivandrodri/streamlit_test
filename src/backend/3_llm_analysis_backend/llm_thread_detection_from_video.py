import json
import os
import time

import streamlit as st
from openai import OpenAI

from backend.utils import encode_image_to_base64
from src.backend.prompts import HAZARD_DETECTION_PROMPT

MAX_TOKENS = 300

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def analyze_frame_with_gpt(frame, model_name):
    """Send frame to GPT-4 Vision for analysis"""
    base64_image = encode_image_to_base64(frame)
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": HAZARD_DETECTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=MAX_TOKENS
        )

        inference_time = time.time() - start_time
        try:
            analysis = json.loads(response.choices[0].message.content)
            return analysis, inference_time
        except json.JSONDecodeError:
            st.error("Failed to parse GPT response as JSON")
            return None, inference_time
    except Exception as e:
        st.error(f"Error calling GPT API: {str(e)}")
        return None, 0
