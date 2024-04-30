import json as json
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import requests
from PIL import Image
from io import BytesIO
from hashlib import sha256
from .encoders import get_sentence_embedding, get_image_embedding

class Images(BaseModel):
    url: str
    embedding: Optional[List[float]] = None

    def get_image(self, timeout: int = 4) -> Optional[Image.Image]:
        try:
            response = requests.get(self.url, timeout=timeout)
            response.raise_for_status()  
            return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            print(f"Failed to retrieve image from {self.url}. Error: {e}")
            return None


    def __init__(self, **data):

        super(Images, self).__init__(**data)

        if self.embedding is None:
            img = self.get_image()
            
            if img is  None:
                return
        
            try:
                self.embedding = get_image_embedding(img)
            except Exception as e:
                print("Error while encoding images ", e)
                

class Tools(BaseModel):
    displayName: str
    images: List[Images]
    embedding: List[float] = list()

    def __init__(self, **data):

        super(Tools, self).__init__(**data)

        if len(self.embedding) == 0:
            self.embedding: List[float] = get_sentence_embedding(self.displayName)


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
            self.embedding: List[float] = get_sentence_embedding(text)


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
            self.embedding: List[float] = get_sentence_embedding(text)


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
            self.embedding: List[float] = get_sentence_embedding(text)


def create_Recipy(data: Dict[str, Any]):

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
        n_hits=min(n_hits, size),
        hits=[QueryHit(score=i["_score"], recipe=Recipe(**i["_source"])) for i in hits],
    )
