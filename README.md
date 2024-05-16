# Cooking Conversational Bot

## Web and Data Mining 2024 Project

Structure:

example_queries.ipynb holds our queries with examples

data_formats.py - Dataformats Pydantic Models and Dual Encoder Embedding Function
save_data.py - Convert into pydantic models with embeddning and save as json
client.py - holds opensearch client
fill_index.py - Fills our open search index
queries.py - Holds our queries

## Install

```
pip install -r requirements.txt¸

pip install -e .

optimum-cli export onnx -m sentence-transformers/all-MiniLM-L6-v2 --optimize O2 models/all-MiniLM-L6-v2

optimum-cli export onnx -m deepset/roberta-base-squad2 --task question-answering --optimize O2 models/roberta-base-squad2

optimum-cli export onnx -m models/twiz-intent-model --task text-classification --optimize O2 models/twiz-intent```