import random
from abc import abstractmethod, ABCMeta
from enum import StrEnum, EnumMeta
from typing import List, Dict, Tuple, Callable
import time

import numpy as np
from openai import OpenAI
import fireworks.client
import json
from pydantic import BaseModel, Field
from src.backend.factories.api_keys_factory import FactoryKey
from src.backend.factories.llm_messages_factory import LlmMessageFactory
from src.backend.logging_config import log_llm_usage

MAX_TOKENS = 100


class ModelProviders(StrEnum):
    open_ai = "OPEN_AI"
    fireworks = "FIREWORKS"
    fake_provider = "FAKE_PROVIDER"


class LLMInput(BaseModel):
    model_name: str
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
        return ModelProviders.open_ai

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
    firellava_13b = "accounts/ivandiegorodriguez-a46102/deployedModels/firellava-13b-9147df42"

    def get_response(self, model_input_data: LLMInput) -> LlmOutput:
        match self:
            case self.Llama_3_11B_Vision | self.Llama_3_90B_Vision | self.Phi_3_Vision | self.firellava_13b:
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
            case self.firellava_13b:
                return {"token_price": 0.20, "context": 4000}

    def get_total_cost(self, num_input_tokens: int, num_output_tokens: int) -> float:
        cost_tokens = (num_input_tokens * self.get_euro_cost_per_million_tokens()["token_price"] +
                       num_output_tokens * self.get_euro_cost_per_million_tokens()["token_price"]) / 10.0 ** 6
        return cost_tokens

    @classmethod
    def get_provider_name(cls) -> str:
        return ModelProviders.fireworks

    def admit_batch_mode(self) -> bool:
        match self:
            case self.Llama_3_11B_Vision | self.Llama_3_90B_Vision | self.Phi_3_Vision | self.firellava_13b:
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
        return ModelProviders.fake_provider

    def admit_batch_mode(self) -> bool:
        return True


class LlmModelFactory(StrEnum):
    fake_model = "fake_model"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o = "gpt-4o"
    Llama_3_11B_Vision = "Llama_3_11B_Vision"
    Llama_3_90B_Vision = "Llama_3_90B_Vision"
    Phi_3_Vision = "Phi_3_Vision"
    firellava_13b = "firellava_13b"

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
            case self.firellava_13b:
                return FireworkModelFactory.firellava_13b
            case self.fake_model:
                return FakeModelFactory.fake_model
            case _:
                raise ValueError(f"The model {self} is not yet registered")

    def get_response(
            self,
            prompt: str,
            frames: np.ndarray | List[np.ndarray] | None = None,
            batch_mode: bool = False,
            max_num_tokens: int = 200,
            min_response_time_fake_model: float | None = None,
            max_response_time_fake_model: float | None = None
    ) -> List[LlmOutput]:
        match self:
            case self.gpt_4o | self.gpt_4o_mini | self.fake_model:
                message = LlmMessageFactory.multi_frame_message_openai.get_message if batch_mode else (
                    LlmMessageFactory.single_frame_message_openai.get_message)
            case self.Llama_3_11B_Vision | self.Llama_3_90B_Vision | self.Phi_3_Vision | self.firellava_13b:
                message = LlmMessageFactory.single_frame_message_fireworks.get_message
            case _:
                raise ValueError(f"No message generator from LlmMessageFactory assign to model {self.value}")

        if batch_mode and not self.get_llm_model().admit_batch_mode():
            response = get_llm_response_for_fake_batch(
                model_name=self.name,
                frames=frames,
                prompt=prompt,
                message_generator=message,
                max_num_tokens=max_num_tokens,
                min_response_time_fake_model=min_response_time_fake_model,
                max_response_time_fake_model=max_response_time_fake_model
            )
        else:
            response = get_llm_response_for_batch(
                model_name=self.name,
                prompt=prompt,
                frames=frames,
                message_generator=message,
                max_num_tokens=max_num_tokens,
                min_response_time_fake_model=min_response_time_fake_model,
                max_response_time_fake_model=max_response_time_fake_model
            )
        return [response]

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


def get_llm_response_for_batch(
        model_name: str,
        prompt: str,
        message_generator: Callable,
        frames: List[np.ndarray] | np.ndarray | None = None,
        max_num_tokens: int = 200,
        min_response_time_fake_model: float | None = None,
        max_response_time_fake_model: float | None = None
):
    if not (isinstance(frames, np.ndarray) or isinstance(frames, List)):
        raise ValueError(f"In batch mode frames must be of type np.ndarray or List[np.ndarray]")

    message = message_generator(prompt=prompt, frames=frames)
    model_input = LLMInput(
        model_name=model_name,
        message=message,
        max_tokens=max_num_tokens,
        fake_inference_time_range=(min_response_time_fake_model, max_response_time_fake_model)
    )

    llm_model = LlmModelFactory[model_name].get_llm_model()

    llm_response = llm_model.get_response(model_input)
    return llm_response


def _aggregate_llm_outputs_in_fake_batch_mode(llm_outputs: List[LlmOutput]) -> LlmOutput:

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


def get_llm_response_for_fake_batch(
        model_name: str,
        prompt: str,
        message_generator: Callable,
        frames: List[np.ndarray] | None = None,
        max_num_tokens: int = 200,
        min_response_time_fake_model: float | None = None,
        max_response_time_fake_model: float | None = None
):
    llm_model = LlmModelFactory[model_name].get_llm_model()
    responses = []
    llm_with_context_outputs = []
    llm_cost = 0.0
    llm_latency = 0.0
    for idx, frame in enumerate(frames):

        if idx == 0:
            prompt_with_context = f"Frame {idx + 1}:\n{prompt}"
        else:
            prompt_with_context = f"""
                       Previous analysis: {responses[-1]}

                       Now analyze the next frame (Frame {idx + 1}):
                       {prompt}
                       """

        message = message_generator(
            frames=frame,
            prompt=prompt_with_context,
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
        llm_latency += llm_response.inference_time
        llm_with_context_outputs.append(llm_response)

    llm_response = _aggregate_llm_outputs_in_fake_batch_mode(llm_with_context_outputs)
    llm_response.cost = llm_cost
    llm_response.inference_time = llm_latency
    return llm_response
