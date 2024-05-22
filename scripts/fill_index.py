from cooking_bot.data_formats import *
from tqdm import tqdm
from cooking_bot.client import INDEX_NAME, CLIENT
from cooking_bot.encoders import SENTENCE_TRANSFORMER_DIM, CLIP_DIM
from glob import glob
import pydantic

sentence_embedding = {
    "type": "knn_vector",
    "dimension": SENTENCE_TRANSFORMER_DIM,
    "method": {
        "name": "hnsw",
        "space_type": "innerproduct",
        "engine": "faiss",
        "parameters": {"ef_construction": 256, "m": 48},
    },
}


image_dict = {
    "type": "nested",
    "properties": {
        "url": {"type": "keyword"},
        "embedding": {
            "type": "knn_vector",
            "dimension": CLIP_DIM,
            "method": {
                "name": "hnsw",
                "space_type": "innerproduct",
                "engine": "faiss",
                "parameters": {"ef_construction": 256, "m": 48},
            },
        },
        "bytes": {"type": "binary"},
    },
}


bm25_standard = {
    "type": "text",
    "analyzer": "standard",
    "similarity": "BM25",
}
text_standard = {"type": "text", "analyzer": "standard"}


index_body = {
    "settings": {
        "index": {
            "number_of_replicas": 0,
            "number_of_shards": 4,
            "refresh_interval": "1s",
            "knn": True,
        }
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "displayName": bm25_standard,
            "description": bm25_standard,
            "tools": {
                "type": "nested",
                "properties": {
                    "displayName": text_standard,
                    "images": image_dict,
                    "embedding": sentence_embedding,
                },
            },
            "ingredients": {
                "type": "nested",
                "properties": {
                    "displayText": text_standard,
                    "ingredient": {"type": "keyword"},
                    "ingredientId": {"type": "keyword"},
                    "quantity": {"type": "float"},
                    "unit": {"type": "keyword"},
                    "images": image_dict,
                    "embedding": sentence_embedding,
                },
            },
            "images": image_dict,
            "instructions": {
                "type": "nested",
                "properties": {
                    "stepNumber": {"type": "integer"},
                    "stepTitle": text_standard,
                    "stepText": text_standard,
                    "stepImages": image_dict,
                    "embedding": sentence_embedding,
                },
            },
            "totalTimeMinutes": {"type": "integer"},
            "embedding": sentence_embedding,
        },
    },
}


with open("index_body.json", "w") as f:
    json.dump(index_body, f, indent=4)


if CLIENT.indices.exists(INDEX_NAME):

    print("delete existing")
    response = CLIENT.indices.delete(index=INDEX_NAME, timeout=10)


CLIENT.indices.create(index=INDEX_NAME, body=index_body)


for id, path in enumerate(tqdm(glob("jsons/*.json"), "Filling Index")):
    
    with open(path, "r") as f:

        data = f.read()
    
    recipe = Recipe.model_validate_json(data)
    
    

    response = CLIENT.index(index=INDEX_NAME, body=recipe.model_dump(), id=id)
    if response["result"] != "created":
        print("Failure", response)

CLIENT.indices.close(index=INDEX_NAME)
