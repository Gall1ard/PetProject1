import openai
from config import load_config, Config, InvalidLLMConfigurationError
from openai import OpenAI
import re
import json
import os
from predict import Prediction


def stringify_prediction(pred_results: Prediction) -> str:
    pred2dict = pred_results.to_dict()

    return json.dumps(
            pred2dict,
            ensure_ascii=False,
            indent=2
        )

def check_file(path):
    return (
        os.path.exists(path)
        and os.path.isfile(path)
        and os.path.getsize(path) > 0
    )

def extract_json_data(mixed_string: str | None) -> dict:
    if not mixed_string:
        return {}

    mixed_string = re.sub(r"^```json\s*", "", mixed_string.strip())
    mixed_string = re.sub(r"\s*```$", "", mixed_string)

    try:
        return json.loads(mixed_string)
    except json.JSONDecodeError:
        return {}


def create_client(conf: Config) -> OpenAI:
    return OpenAI(
        api_key=conf.api_key,
        base_url=conf.api_base
    )


def get_reasoning(prediction: Prediction) -> str | None:
    conf = load_config()

    # If LLM is not connected
    if not conf.llm_enabled():
        raise InvalidLLMConfigurationError(
            "LLM configuration not enabled."
        )

    # If LLM is connected
    client = create_client(conf)
    system_prompt = ""
    messages = []

    if not prediction.is_valid():
        raise ValueError(
            f"Prediction is invalid: {prediction.to_dict()}"
        )

    pred_results_str = stringify_prediction(prediction)

    if not check_file(conf.system_prompt_path):
        raise FileNotFoundError(
            f"System prompt not found: {conf.system_prompt_path}"
        )

    with open(conf.system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    if system_prompt.strip():
        messages.append(
            {"role": "system", "content": system_prompt}
        )

    if pred_results_str.strip():
        messages.append(
            {"role": "user", "content": pred_results_str}
        )

    try:
        response = client.chat.completions.create(
            model=conf.llm,
            messages=messages,
        )
        llm_output_str = response.choices[0].message.content
        llm_output_dict = extract_json_data(llm_output_str)


        return f"{llm_output_dict.get(
            "reasoning",
            "Reasoning unavailable"
        )}"

    except openai.OpenAIError as e:
        raise e