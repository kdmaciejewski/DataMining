import os
import requests
import json


internal_url = 'http://10.10.255.202:5633'
external_url = "https://twiz.novasearch.org/"
max_timeout = 10

def test_ping(url: str):
    # Make a GET request to the URL
    response = requests.get(url, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("GET request successful for URL:", url)
        print("Response:", response.text)
    else:
        print("GET request failed with status code:", response.status_code)


def test_raw_post_request(url: str):

    url = os.path.join(url, "raw")

    # This is just an example that you can try to play around
    # test_text = "Hi how are you today?"
    # In practice it should be in a format similar to this
    # test_text = "<|prompter|> You are a taskbot tasked with helping users cook recipes or DIY projects. 
    # I will give you a recipe and I want you to help me do it step by step. You should always be empathetic, 
    # honest, and should always help me. If I ask you something that does not relate to the recipe you should politely reject the request 
    # and try too get me focused on the recipe. I am unsure how to cook something or do something related to the recipe you should 
    # help me to the best of your ability. Please use a neutral tone of voice. 
    # Recipe: Test Recipe Steps: Step 1: Preheat oven to 350 degrees 
    # Step 2: Mix ingredients together 
    # Step 3: Bake for 30 minutes <|endofturn|> <|prompter|> I haven't started cooking yet. <|endofturn|> <|assistant|> ok! 
    # <|endofturn|> <|prompter|> Hello <|endofturn|> <|assistant|>"


    # test_text = "<|prompter|> You are a taskbot tasked with helping users cook recipes or DIY projects. I will give you a recipe and I want you to help me do it step by step. You should always be empathetic, honest, and should always help me. If I ask you something that does not relate to the recipe you should politely reject the request and try too get me focused on the recipe. I am unsure how to cook something or do something related to the recipe you should help me to the best of your ability. Please use a neutral tone of voice. Recipe: Test Recipe Steps: Step 1: Preheat oven to 350 degrees Step 2: Mix ingredients together Step 3: Bake for 30 minutes <|endofturn|> <|prompter|> I haven't started cooking yet. <|endofturn|>" # <|assistant|> ok! <|endofturn|> <|prompter|> Hello <|endofturn|> <|assistant|>"
    test_text = "You are a taskbot tasked with helping users cook recipes or DIY projects. \
        I will give you a recipe and I want you to help me do it step by step. You should always be \
        empathetic, honest, and should always help me. If I ask you something that does not relate\
        to the recipe you should politely reject the request and try too get me focused on the recipe.\
        I am unsure how to cook something or do something related to the recipe you should help me to \
        the best of your ability. Please use a neutral tone of voice."

    
    
    data = {
        "text": test_text,
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": -1,
    }

    # Make the POST request
    response = requests.post(url, json=data, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("POST request successful for URL:", url)
        print("Response:", response.text)
    else:
        print("Response:", response.text)


def test_structured_post_request(url: str):

    url = os.path.join(url, "structured")

    # check this file to understand the structure of the data
    with open('project\src\cooking_bot\example_conversation.json') as f:
        data = json.load(f)

    data = {
        "dialog": data,
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": -1,
    }

    # Make the POST request
    response = requests.post(url, json=data, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("POST request successful for URL:", url)
        formatted_response = response.text.encode().decode('unicode_escape')
        print("Response:", formatted_response)
    else:
        print("POST request failed with status code:", response.status_code)
        print("POST request failed with status code:", response.status_code)


def send_message(url: str, message: str):
    url = os.path.join(url, "raw")

    data = {
        "text": message,
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": -1,
    }

    # Make the POST request
    response = requests.post(url, json=data, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # return response.json().get("text", "")
        return response.text
    else:
        print("POST request failed with status code:", response.status_code)
        return None


def test_chat(url: str):
    
    with open('project\src\cooking_bot\example_conversation.json') as f:
        data = json.load(f)
        
    system_tone = data["system_tone"] 
    recipe_name = data["task"]["recipe"]["displayName"]
    instructions = data["task"]["recipe"]["instructions"]
    instruction_string = ""
    
    for i, step in enumerate(instructions, 1):
        instruction_string += f"Step {i}: {step['stepText']}\n"
    
    print(instruction_string)
    setup_prompt = f"Now you will get an instructions to manage conversation about a recipe: {recipe_name}. Your tone should be {system_tone}\
            When a user asks you about the next step you will take the next element from the steps list. Here is the steps list: {instruction_string}\
            I want you to give only one step at a time and the answers should be short and based on the steps that I provided to you\
            the user doesn't know the recipe so include the informations from this prompt in your answers. Answers must be short and don't include questions."
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting conversation...")
            break
        
        # user_input += " Answers must be connected to the cooking recipe and try to keep them short and not include unnecessary steps."
        # Send user message and get model response
        model_response = send_message(url, user_input)
        if model_response:
            print("Model:", model_response)
        else:
            print("Failed to get model response.")

         
if __name__ == '__main__':

    # ping
    # test_ping(internal_url)
    # test_ping(external_url)

    # raw
    # test_raw_post_request(internal_url)
    # test_raw_post_request(external_url)

    # strutured
    # test_structured_post_request(internal_url)
    test_structured_post_request(external_url)

    test_chat(external_url)