import os
import requests
from cooking_bot.data_formats import *
import os
from pydantic import BaseModel
import time
from loguru import logger


class PromptSettings(BaseModel):
    max_tokens : int = 100
    temperature : float =  0.0
    top_p : int =  1
    top_k : int =  3
    

URL = os.environ["PLAN_LLM_URL"]

def test_ping(timeout = 10):
    start = time.time()
    response = requests.get(URL, timeout=timeout)
    response.raise_for_status()
    logger.success(f"Api is active. Ping is {time.time() - start :.2f}")


def send_message( message: str, settings : PromptSettings, timeout = 10):
    url = os.path.join(URL, "raw")

    data = settings.model_dump()
    data["text"] = message

    # Make the POST request
    response = requests.post(url, json=data, timeout=timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # return response.json().get("text", "")
        return response.text
    else:
        print("POST request failed with status code:", response.status_code)
        return None

