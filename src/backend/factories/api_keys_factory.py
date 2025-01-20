import os
from enum import StrEnum
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../../.env")


class FactoryKey(StrEnum):
    open_ai = "open_ai"
    fireworks = "fireworks"

    def get_key(self):

        match self:
            case self.open_ai:
                if not os.getenv('OPENAI_API_KEY'):
                    raise ValueError("OPENAI_API_KEY not set in .env")
                return os.getenv('OPENAI_API_KEY')
            case self.fireworks:
                if not os.getenv('FIREWORKS_API_KEY'):
                    raise ValueError("FIREWORKS_API_KEY not set in .env")
                return os.getenv('FIREWORKS_API_KEY')
            case _:
                raise ValueError(f"Key {self} is not supported")
