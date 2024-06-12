# Cooking Conversational Bot

## Overview
This repository contains the Cooking Conversational Bot, a project developed for the Web and Data Mining 2024 course at Universidade Nova de Lisboa. The bot leverages OpenSearch and advanced NLP models to assist users in cooking by providing recipe suggestions and guidance based on user queries.

## Notable Files
- [`conversation.py`](src/cooking_bot/conversation.py) - State management and entry points; the main script.
- [`intent_regex.py`](src/cooking_bot/intent_regex.py) - Regex cascade for intent classification.
- [`intent_detector.py`](src/cooking_bot/intent_detector.py) - Our intent detector assembled.
- [`prompt.txt`](datasets/prompt.txt) - Prompt used to generate our recipe intent dataset.
- [`recipe_intent_dataset.ipynb`](notebooks/recipe_intent_datsaset.ipynb) - Cleaning the recipe intent data.
- [`train_recipe_classifiers.py`](scripts/train_recipe_classifiers.py) - Train recipe intent classifiers, test them, and save models.
- [`example_queries.ipynb`](notebooks/example_querries.ipynb) - Jupyter notebook containing sample queries and their outputs.
- [`queries.py`](src/cooking_bot/queries.py) - Contains predefined queries used by the conversational bot.
- [`data_formats.py`](src/cooking_bot/data_formats.py) - Defines Pydantic models for data validation and serialization, and includes a function for dual encoder embeddings.
- [`save_data.py`](scripts/save_data.py) - Scripts to convert data into Pydantic models, embed using dual encoders, and save as JSON.
- [`fill_index.py`](scripts/fill_index.py) - Script to populate the OpenSearch index with recipe data.

## Installation
Before running the bot, you need to install the necessary dependencies and set up the models. Follow these steps:

### Install Python Packages:
```
pip install -r requirements.txt
pip install optimum[exporters]
pip install -e .
```

### Setting Up Environment Variables

To run the Cooking Conversational Bot, you need to configure your environment variables. We use a `.env` file to securely manage these settings.

```
NOVA_SEARCH_US=user205 # our OpenSearch username
NOVA_SEARCH_PW=password # Password for the OpenSearch index user
NOVA_SEARCH_HOST=your_opensearch_host # Host URL of the OpenSearch index
NOVA_SEARCH_PORT=your_opensearch_port # Port for the OpenSearch index
PLAN_LLM_URL=your_plan_llm_url # URL to the Plan LLM API
```

### Exporting Models to ONNX Runtime

Since we did not have gpus in our laptops, we optimized the models using onnx. Therefore you need to export these models to onnx following these commands.

```
optimum-cli export onnx -m sentence-transformers/all-MiniLM-L6-v2 --optimize O2 models/all-MiniLM-L6-v2

optimum-cli export onnx -m deepset/roberta-base-squad2 --task question-answering --optimize O2 models/roberta-base-squad2

optimum-cli export onnx -m models/twiz-intent-model --task text-classification --optimize O2 models/twiz-intent```
```
Note that you the last command must be adjusted to the location of the twiz-intent-model.

Then simply run the command ```cooking-bot```, which is exposed as part of our python package install or run src/cooking_bot/conversation.py.