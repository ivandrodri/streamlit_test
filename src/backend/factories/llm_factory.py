import random
from abc import abstractmethod, ABCMeta
from enum import StrEnum, EnumMeta
from typing import List, Dict, Tuple
import time
import numpy as np
from openai import OpenAI
import fireworks.client
import json
from pydantic import BaseModel, Field
from src.backend.factories.api_keys_factory import FactoryKey
from src.backend.factories.logging_config import log_llm_usage

MAX_TOKENS = 100


class LLMInput(BaseModel):
    model_name: str
    prompt: str | None = None# ToDo: I don't need this here anymore in current version --> used to create message
    frames: np.ndarray | List[np.ndarray] | None = None  # ToDo: I don't need this here anymore in current version --> used to create message
    message: List[Dict] | str = Field(default_factory=list)
    max_tokens: int = MAX_TOKENS  # Replace with actual MAX_TOKENS if defined elsewhere
    response_format: Dict = {"type": "json_object"}
    seed: int = None
    fake_inference_time_range: Tuple[float, float] | None = None

    class Config:
        arbitrary_types_allowed = True


class LlmOutput(BaseModel):
    response: str | Dict | None = None
    inference_time: float | None = None
    cost: float | None = None


class ABCEnumMeta(ABCMeta, EnumMeta):
    pass


class BaseModelFactory(StrEnum, metaclass=ABCEnumMeta):

    @abstractmethod
    def get_response(self, model_input_data) -> 'LlmOutput':
        pass

    @abstractmethod
    def get_euro_cost_per_million_tokens(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_total_cost(self, num_input_tokens: int, num_output_tokens: int) -> float:
        pass

    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        pass

    @classmethod
    def get_model_names(cls) -> List[str]:
        return [elem for elem in cls.__members__.keys()]

    @abstractmethod
    def admit_batch_mode(self) -> bool:
        pass


class OpenAIModelFactory(BaseModelFactory):
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o = "gpt-4o"

    # ToDo: The function below repeat in firework implementation too so could be absorbed in the base class.
    def get_response(self, model_input_data: LLMInput) -> LlmOutput:
        match self:
            case self.gpt_4o_mini | self.gpt_4o:
                completions = OpenAI(api_key=FactoryKey.open_ai.get_key()).chat.completions
                try:
                    start_time = time.time()
                    response = completions.create(
                        model=self,
                        messages=model_input_data.message,
                        max_tokens=model_input_data.max_tokens,
                        response_format=model_input_data.response_format,
                        #seed=model_input_data.seed,
                    )
                    inference_time = time.time() - start_time

                    cost = self.get_total_cost(
                            num_input_tokens=response.usage.prompt_tokens,
                            num_output_tokens=response.usage.completion_tokens
                    )

                    log_llm_usage(
                        cost=cost,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens
                    )

                    return LlmOutput(
                        response=json.loads(response.choices[0].message.content),
                        inference_time=inference_time,
                        cost=cost,
                    )

                except Exception as e:
                    raise Exception(f"OpenAI API error: {str(e)}")

            case _:
                raise ValueError(f"The model you try to create {self} is not yet registered.")

    def get_euro_cost_per_million_tokens(self) -> Dict[str, float]:
        match self:
            case self.gpt_4o:
                return {"token_price_input": 5, "token_price_output": 15}
            case self.gpt_4o_mini:
                return {"token_price_input": 0.15, "token_price_output": 0.60}
            case _:
                raise ValueError(f"The model you try to access {self} is not yet registered.")

    def get_total_cost(self, num_input_tokens: int, num_output_tokens: int):
        cost_tokens = (num_input_tokens*self.get_euro_cost_per_million_tokens()["token_price_input"] +
                       num_output_tokens*self.get_euro_cost_per_million_tokens()["token_price_output"])/10.0**6
        return cost_tokens

    @classmethod
    def get_provider_name(cls) -> str:
        return "OPENAI"

    def admit_batch_mode(self) -> bool:
        match self:
            case self.gpt_4o | self.gpt_4o_mini:
                return True
            case _:
                raise ValueError(f"The model you try to access {self} is not yet registered.")


class FireworkModelFactory(BaseModelFactory):

    # ToDo: IMPORTANT --> Replace values by real model names
    Llama_3_11B_Vision = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
    Llama_3_90B_Vision = "accounts/fireworks/models/llama-v3p2-90b-vision-instruct"
    Phi_3_Vision = "accounts/fireworks/models/phi-3-vision-128k-instruct"  # 4B params

    def get_response(self, model_input_data: LLMInput) -> LlmOutput:
        match self:
            case self.Llama_3_11B_Vision | self.Llama_3_90B_Vision | self.Phi_3_Vision:
                fireworks.client.api_key = FactoryKey.fireworks.get_key()

                completions = fireworks.client.ChatCompletion
                try:
                    start_time = time.time()
                    response = completions.create(
                        model=self,
                        messages=model_input_data.message,
                        max_tokens=model_input_data.max_tokens,
                        response_format=model_input_data.response_format,
                        #seed=model_input_data.seed
                    )
                    inference_time = time.time() - start_time

                    content = response.choices[0].message.content

                    # ToDo: Why is this needed --> encapsulate in a preprocessing function
                    # Remove any leading/trailing whitespace and handle potential string formatting
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:-3]  # Remove ```json and ``` markers
                    elif content.startswith('{'):
                        content = content  # Already clean JSON

                    return LlmOutput(
                        response=json.loads(content),
                        inference_time=inference_time,
                        cost=self.get_total_cost(
                            num_input_tokens=response.usage.prompt_tokens,
                            num_output_tokens=response.usage.completion_tokens
                        )
                    )

                except Exception as e:
                    raise Exception(f"Fireworks API error: {str(e)}")

    def get_euro_cost_per_million_tokens(self) -> Dict[str, float]:
        match self:
            case self.Llama_3_90B_Vision:
                return {"token_price": 0.90, "context": 128000}
            case self.Llama_3_11B_Vision:
                return {"token_price": 0.20, "context": 128000}
            case self.Phi_3_Vision:
                return {"token_price": 0.20, "context": 31000}

    def get_total_cost(self, num_input_tokens: int, num_output_tokens: int) -> float:
        cost_tokens = (num_input_tokens * self.get_euro_cost_per_million_tokens()["token_price"] +
                       num_output_tokens * self.get_euro_cost_per_million_tokens()["token_price"]) / 10.0 ** 6
        return cost_tokens

    @classmethod
    def get_provider_name(cls) -> str:
        return "FIREWORKS"

    def admit_batch_mode(self) -> bool:
        match self:
            case self.Llama_3_11B_Vision | self.Llama_3_90B_Vision | self.Phi_3_Vision:
                return False
            case _:
                raise ValueError(f"The model you try to access {self} is not yet registered.")


class FakeModelFactory(BaseModelFactory):
    fake_model = "fake_model"

    @staticmethod
    def fake_llm_output(
            min_response_time: float = 2.5,
            max_response_time: float = 3.5
    ) -> LlmOutput:
        llm_output = LlmOutput()

        hazard_types = ["theft", "robbery", "none"]
        hazard_detected = [True, False]
        llm_output.response = {
            "hazard_detected": random.choice(hazard_detected),
            "hazard_type": random.choice(hazard_types),
            "reasoning": "Hello This is a test response from a fake agent"
        }
        llm_output.cost = 0.0
        time_rnd = random.uniform(min_response_time, max_response_time)
        time.sleep(time_rnd)
        llm_output.inference_time = time_rnd
        return llm_output

    def get_response(self, model_input_data: LLMInput) \
            -> 'LlmOutput':
        match self:
            case self.fake_model:

                if model_input_data.fake_inference_time_range is None:
                    raise ValueError(f"You must provide the fake inference range for {self.fake_model}.")

                return self.fake_llm_output(
                        min_response_time=model_input_data.fake_inference_time_range[0],
                        max_response_time=model_input_data.fake_inference_time_range[1]
                )
            case _:
                raise ValueError(f"The fake model {self} is not yet registered.")

    def get_euro_cost_per_million_tokens(self) -> Dict[str, float]:
        return {"token_price": 0.0, "context": 0}

    def get_total_cost(self, num_input_tokens: int, num_output_tokens: int) -> float:
        return 0.0

    @classmethod
    def get_provider_name(cls) -> str:
        return "FAKE_PROVIDER"

    def admit_batch_mode(self) -> bool:
        return True


class LlmModelFactory(StrEnum):
    fake_model = "fake_model"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o = "gpt-4o"
    Llama_3_11B_Vision = "Llama_3_11B_Vision"
    Llama_3_90B_Vision = "Llama_3_90B_Vision"
    Phi_3_Vision = "Phi_3_Vision"

    def get_llm_model(self):
        match self:
            case self.gpt_4o_mini:
                return OpenAIModelFactory.gpt_4o_mini
            case self.gpt_4o:
                return OpenAIModelFactory.gpt_4o
            case self.Llama_3_11B_Vision:
                return FireworkModelFactory.Llama_3_11B_Vision
            case self.Phi_3_Vision:
                return FireworkModelFactory.Phi_3_Vision
            case self.Llama_3_90B_Vision:
                return FireworkModelFactory.Llama_3_90B_Vision
            case self.fake_model:
                return FakeModelFactory.fake_model
            case _:
                raise ValueError(f"The model {self} is not yet registered")

    @classmethod
    def get_model_names(cls) -> List[str]:
        return [elem for elem in cls.__members__.keys()]


'''
import base64
from google.protobuf.message_factory import MessageFactory
from backend.prompts import HAZARD_DETECTION_GRID_PLOT_PROMPT
from backend.factories.llm_messages_factory import LlmMessageFactory
from PIL import Image
from fireworks.client import Fireworks


image_path = '/home/ivan/Documents/GIT_PROJECTS/Guardian_angel_BMW_innovation/GuardianAngel/output_imgs/trial_image_ivan_2.jpg'
image = Image.open(image_path)
image_np = np.array(image)
PROMPT = HAZARD_DETECTION_GRID_PLOT_PROMPT

message = LlmMessageFactory.single_frame_message_fireworks.get_message(prompt=PROMPT, frames=image_np)

fireworks.client.api_key = FactoryKey.fireworks.get_key()
completions = fireworks.client.ChatCompletion
response = completions.create(
    #model="accounts/fireworks/models/llama-v3p2-11b-vision-instruct",  # Replace with your specific model
    model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
    messages=message,
    max_tokens=300,
)

bla = 4
'''
