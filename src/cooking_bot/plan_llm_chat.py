import os
import requests
import json
from cooking_bot.data_formats import *
from cooking_bot import REPO_PATH
from glob import glob
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import time
import random
os.chdir(REPO_PATH)


assert load_dotenv()


with open(random.choice(glob("jsons/*.json"))) as f:

    recipe = Recipe.model_validate_json(f.read())


class PromptSettings(BaseModel):
    max_tokens : int = 100
    temperature : float =  0.0
    top_p : int =  1
    top_k : int =  3
    

settings = PromptSettings()
timeout = 10
URL = os.environ["PLAN_LLM_URL"]

def test_ping():
    start = time.time()
    response = requests.get(URL, timeout=timeout)
    response.raise_for_status()
    print(f"Api is active. Ping is {time.time() - start :.2f}")


def send_message( message: str, settings : PromptSettings, ):
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


def test_chat(recipe: Recipe, settings : PromptSettings):
    recipe_name = recipe.displayName

    system_tone = "neutral"
    instruction_string = "\n".join(
        [f"Step {ins.stepNumber}: {ins.stepText}" for ins in recipe.instructions]
    )
    print("\nThe Recipe Name", recipe_name, "\n", instruction_string, "\n","-"*20)


    prompt = f"<|prompter|>  You are a taskbot tasked with helping users cook recipes or DIY projects. \
        I will give you a recipe and I want you to help me do it step by step. You should always be \
        empathetic, honest, and should always help me. If I ask you something that does not relate\
        to the recipe you should politely reject the request and try too get me focused on the recipe.\
        I am unsure how to cook something or do something related to the recipe you should help me to \
        the best of your ability. Please use a {system_tone} tone of voice. <|endofturn|> <|prompter|>\
        The Recipy is called {recipe_name} and the instruction list is: {instruction_string}.\
        Start with a sentence where you great the similar to Hey, today I will assist you in cooking {recipe_name} how can I help you? <|endofturn|> <|assistant|> "



    while True:
        
        model_response = send_message(prompt, settings=settings)
        
        if model_response is None:
            print("failed to get response")
            break
        
        print("\nModel Response:\n", model_response)
        
        prompt += model_response + " <|endofturn|>"
        
        user_input = input("\nYou:\n")
        
        
        if user_input.lower() == "exit":
            print("Exiting conversation...")
            break

        prompt = prompt + "<|prompter|> " + user_input + " <|endofturn|> <|assistant|> "

if __name__ == "__main__":

    test_ping()
    test_chat(recipe, settings)
