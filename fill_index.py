from data_formats import *
import json
from tqdm import tqdm
from client import INDEX_NAME, CLIENT

with open("data_with_embeddings.json", "r") as f:

    data = json.load(f)

data = [Recipe(**d) for d in data]

EMBEDDING_DIM = len(data[0].embedding)


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
            "displayName": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "BM25",
            },
            "description": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "BM25",
            },
            "tools": {
                "type": "nested",
                "properties": {
                    "displayName": {"type": "text", "analyzer": "standard"},
                    "images": {
                        "type": "nested",
                        "properties": {"url": {"type": "keyword"}},
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "faiss",
                            "parameters": {"ef_construction": 256, "m": 48},
                        },
                    },
                },
            },
            "ingredients": {
                "type": "nested",
                "properties": {
                    "displayText": {"type": "text", "analyzer": "standard"},
                    "ingredient": {"type": "keyword"},
                    "ingredientId": {"type": "keyword"},
                    "quantity": {"type": "float"},
                    "unit": {"type": "keyword"},
                    "images": {
                        "type": "nested",
                        "properties": {"url": {"type": "keyword"}},
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "faiss",
                            "parameters": {"ef_construction": 256, "m": 48},
                        },
                    },
                },
            },
            "images": {"type": "nested", "properties": {"url": {"type": "keyword"}}},
            "instructions": {
                "type": "nested",
                "properties": {
                    "stepNumber": {"type": "integer"},
                    "stepTitle": {"type": "text", "analyzer": "standard"},
                    "stepText": {"type": "text", "analyzer": "standard"},
                    "stepImages": {
                        "type": "nested",
                        "properties": {"url": {"type": "keyword"}},
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "faiss",
                            "parameters": {"ef_construction": 256, "m": 48},
                        },
                    },
                },
            },
            "totalTimeMinutes": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "innerproduct",
                    "engine": "faiss",
                    "parameters": {"ef_construction": 256, "m": 48},
                },
            },
        },
    },
}


with open("index_body.json", "w") as f:
    json.dump(index_body, f, indent=4)


if CLIENT.indices.exists(INDEX_NAME):

    response = CLIENT.indices.delete(index=INDEX_NAME, timeout=10)


CLIENT.indices.create(index=INDEX_NAME, body=index_body)

for id, recipe in enumerate(tqdm(data, "Filling Index")):
    response = CLIENT.index(index=INDEX_NAME, body=recipe.model_dump(), id=id)


CLIENT.indices.close(index=INDEX_NAME)
