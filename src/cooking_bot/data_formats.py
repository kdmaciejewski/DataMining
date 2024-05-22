import json as json
from typing_extensions import Literal
from pydantic import BaseModel, StrictBytes
from typing import List, Dict, Optional, Any
import requests
from PIL import Image
from io import BytesIO
from hashlib import sha256
from .encoders import get_sentence_embedding, get_image_embedding
import base64

IMAGE_SIZE = 400

class Images(BaseModel):
    url: str
    embedding: Optional[List[float]] = list()
    bytes : Optional[str] = None
    
    def __init__(self, *args, **kwargs):
        super(Images, self).__init__(*args, **kwargs)
        
        if self.bytes is None:
            img = self.get_image()
            if img is None:
                return

            img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
            img_byte_arr = BytesIO()
            img.convert("RGB").save(img_byte_arr, format="jpeg")
            self.bytes = base64.b64encode(img_byte_arr.getvalue()).decode("UTF-8")
                
    
    def get_image(self, timeout: int = 4) -> Optional[Image.Image]:
        
        if self.bytes is not None:
            return Image.open(BytesIO(base64.b64decode(self.bytes.encode())))
        
        try:
            response = requests.get(self.url, timeout=timeout)
            response.raise_for_status()  
            return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            print(f"Failed to retrieve image from {self.url}. Error: {e}")
            return None


    def calc_embedding(self):
        if self.embedding:
            return 
        
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



def embed_images(re : Recipe):
    
    
    def embedd_and_clean(imgs : List[Images]):
    
        to_remove = []
        for i,img in enumerate(imgs):
            img.calc_embedding()
            if not img.embedding:
                to_remove.append(i)
        
        if to_remove:
            print(f"Removing {len(to_remove)}  empty embeddings")
        
        return [i for a,i in enumerate(imgs) if a not in to_remove]
            
    re.images = embedd_and_clean(re.images)
            
    for ing in re.instructions:
        ing.stepImages = embedd_and_clean(ing.stepImages)
        

def create_Recipy(data: Dict[str, Any], calc_embeddings = False):

    def recu_remove_null_images(data):

        if "images" in data:
            data["images"] = [i for i in data["images"] if i.get("url") is not None]

        for v in data.values():
            if isinstance(v, dict):
                recu_remove_null_images(v)

    recu_remove_null_images(data)
    
    re = Recipe(**data)
    
    if calc_embeddings:
        embed_images(re)

    return re



class QueryHit(Recipe):

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
        hits=[QueryHit(score=i["_score"], **i["_source"]) for i in hits],
    )
