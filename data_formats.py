import json as json
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import requests
import requests
import PIL
from io import BytesIO
from sentence_transformers import SentenceTransformer
from hashlib import sha256


class DummyModel:

    def encode(self, something):
        return np.array([0.2])


EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# EMBEDDING_MODEL = DummyModel()


def get_embedding(s: str) -> list[float]:

    return EMBEDDING_MODEL.encode(s).tolist()


class Images(BaseModel):
    url: str

    def get_image(self, time_out=4) -> Optional[PIL.Image.Image]:

        if self.url is None:
            return None

        response = requests.get(self.url, timeout=time_out)
        return PIL.Image.open(BytesIO(response.content))


class Tools(BaseModel):
    displayName: str
    images: List[Images]
    embedding: List[float] = list()

    def __init__(self, **data):

        super(Tools, self).__init__(**data)

        if len(self.embedding) == 0:
            self.embedding: List[float] = EMBEDDING_MODEL.encode(
                self.displayName
            ).tolist()


class Ingredient(BaseModel):
    displayText: str
    ingredient: Optional[str]
    ingredientId: str
    quantity: float
    unit: str
    images: List[Images]
    embedding: List[float] = list()

    def __init__(self, **data):

        if data["ingredientId"] is None:
            data["ingredientId"] = sha256(data["displayText"].encode()).hexdigest()

        super(Ingredient, self).__init__(**data)

        text = self.ingredient or self.displayText

        if len(self.embedding) == 0:
            self.embedding: List[float] = EMBEDDING_MODEL.encode(text).tolist()


class Instructions(BaseModel):
    stepNumber: int
    stepTitle: Optional[str]
    stepText: str
    stepImages: List[Images]
    embedding: List[float] = list()

    def __init__(self, **data):

        super(Instructions, self).__init__(**data)

        text = self.stepTitle or self.stepText

        if len(self.embedding) == 0:
            self.embedding: List[float] = EMBEDDING_MODEL.encode(text).tolist()


class Recipe(BaseModel):
    displayName: str
    description: Optional[str]
    tools: List[Tools]
    ingredients: List[Ingredient]
    totalTimeMinutes: Optional[int]
    images: List[Images]
    instructions: List[Instructions]
    embedding: List[float] = list()

    def __init__(self, **data):

        super(Recipe, self).__init__(**data)

        text = self.description or self.displayName

        if len(self.embedding) == 0:
            self.embedding: List[float] = EMBEDDING_MODEL.encode(text).tolist()

def create_Recipy(data : Dict[str, Any]):
    
    def recu_remove_null_images(data):
        
        if "images" in data:
            data["images"] = [i for i in data["images"] if i.get("url") is not None]
            
        
        for v in data.values():
            if isinstance(v, dict):
                recu_remove_null_images(v)


    recu_remove_null_images(data)

    return Recipe(**data)

class QueryHit(BaseModel):

    recipe: Recipe
    score: float


class QueryResult(BaseModel):

    n_hits: int
    hits: List[QueryHit]


def create_QueryResult(dict, size):

    dict = dict.get("hits")
    if dict is None:
        return None

    n_hits = dict.get("total", {}).get("value")

    if n_hits is None:
        return None

    hits = dict.get("hits", [])

    return QueryResult(
        n_hits= min(n_hits, size),
        hits=[QueryHit(score=i["_score"], recipe= Recipe(**i["_source"])) for i in hits],
    )
