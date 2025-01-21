import os
import streamlit as st
from enum import StrEnum
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv(dotenv_path="../../../.env")


class FactoryKey(StrEnum):
    open_ai = "open_ai"
    fireworks = "fireworks"

    def get_key(self):
        match self:
            case self.open_ai:
                key = st.secrets.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
                if not key:
                    raise ValueError("OPENAI_API_KEY not set in secrets or .env")
                return key
            case self.fireworks:
                key = st.secrets.get('FIREWORKS_API_KEY') or os.getenv('FIREWORKS_API_KEY')
                if not key:
                    raise ValueError("FIREWORKS_API_KEY not set in secrets or .env")
                return key
            case _:
                raise ValueError(f"Key {self} is not supported")
