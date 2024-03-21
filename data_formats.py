import json as json
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import requests
import requests
import PIL
from io import BytesIO
from sentence_transformers import SentenceTransformer


class DummyModel:
    
    def encode(self, something):
        return np.array([0.2])

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# EMBEDDING_MODEL = DummyModel()

class Images(BaseModel):
    url: Optional[str]
    
    def get_image(self, time_out = 4) -> Optional[PIL.Image.Image]:
        
        if self.url is None:
            return None
        
        response = requests.get(self.url, timeout = time_out)
        return PIL.Image.open(BytesIO(response.content))
    
        

class Tools(BaseModel):
    displayName : str
    images : List[Images]
    embedding : List[float] = list()
    
    def __init__(self, **data):
        
        
        super(Tools, self).__init__( **data)
        
        if len(self.embedding) == 0:
            self.embedding : List[float] = EMBEDDING_MODEL.encode(self.displayName).tolist()
        
    

class Ingredient(BaseModel):
    displayText : str
    ingredient: Optional[str]
    ingredientId: Optional[str]
    quantity : float
    unit : str
    images : List[Images]
    embedding : List[float] = list()
    
    def __init__(self, **data):
        
        
        super(Ingredient, self).__init__( **data)
        
        text = self.ingredient or self.displayText
        
        if len(self.embedding) == 0:
            self.embedding : List[float] = EMBEDDING_MODEL.encode(text).tolist()
       
       
class Instructions(BaseModel):
    stepNumber : int
    stepTitle : Optional[str]
    stepText : str
    stepImages : List[Images]
    embedding : List[float] = list()
    
    
    def __init__(self, **data):
        
        super(Instructions, self).__init__( **data)
        
        text = self.stepTitle or self.stepText
        
        if len(self.embedding) == 0:
            self.embedding : List[float] = EMBEDDING_MODEL.encode(text).tolist()
            
        
    
class Recipe(BaseModel):
    displayName : str
    description: Optional[str]
    tools: List[Tools]
    ingredients: List[Ingredient]
    images : List[Images]
    instructions : List[Instructions]
    embedding : List[float] = list()
    
    
    def __init__(self, **data):
        
        
        super(Recipe, self).__init__( **data)
        
        text = self.description or self.displayName
        
        if len(self.embedding) == 0:
            self.embedding : List[float] = EMBEDDING_MODEL.encode(text).tolist()
        
    
    