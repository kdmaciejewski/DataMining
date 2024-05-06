import json as json
import PIL.Image
from typing import List
import numpy as np
import PIL
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
from abc import ABC, abstractmethod
import time
from PIL import Image
import requests

SENTENCE_TRANSORMER = "all-MiniLM-L6-v2" 
CLIP_MODEL = "openai/clip-vit-base-patch32"

class DummyModel:

    @classmethod
    def encode(cls, something):
        return np.array([0.2])


class LazyModel(ABC):
    
    instance = None
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def encode_text(self, txt : str) -> List[float]:
        pass
    
    
    @classmethod 
    def get_instance(cls):
        
        if cls.instance is None: 
            cls.instance = cls()
            
        return cls.instance
    
    def to_normed_list(self, emb : torch.TensorType) -> List[float]:
        return torch.nn.functional.normalize(emb, dim = -1).numpy().tolist()
    
    
    
    @classmethod
    def lazy_encode_text(cls, text : str)  -> list[float]:
        return cls.get_instance().encode_text(text)
    
    @classmethod 
    def get_dim(cls):
        return len(cls.get_instance().encode_text("bla bla"))
        
        
        
        
        

class LazySentenceTransformer(LazyModel):
    def __init__(self) -> None:
        self.model = SentenceTransformer(SENTENCE_TRANSORMER)
        
    
    def encode_text(self, text : str)  -> list[float]:
        
        emb = self.model.encode(text, convert_to_numpy=False, convert_to_tensor=True)
        return self.to_normed_list(emb)
        
    
    

    
class LazyClipModel(LazyModel):

    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL)
    
    
    def encode_text(self, text : str):
        inputs = self.processor(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        return self.to_normed_list(text_features[0])
    
    
    def encode_img(self, img : PIL.Image):
        inputs = self.processor(images = img, return_tensors="pt", padding=True)

        with torch.no_grad():
            img_features = self.model.get_image_features(**inputs)
            
        return self.to_normed_list(img_features[0])
    
    
    @classmethod 
    def lazy_encode_img(cls, img : PIL.Image)  -> list[float]:
        return cls.get_instance().encode_img(img)
    
        
# get_sentence_embedding = DummyModel.encode

get_sentence_embedding = LazySentenceTransformer.lazy_encode_text
get_image_embedding = LazyClipModel.lazy_encode_img
get_clip_text_embedding = LazyClipModel.lazy_encode_text

SENTENCE_TRANSFORMER_DIM = 384
CLIP_DIM = 512


if __name__ == "__main__":
    
    
    sentence = "Hello, world!"
    
    start_time = time.time()
    sentence_embedding = get_sentence_embedding(sentence)
    first_call_duration = time.time() - start_time
    print(f"First call: Embedding for '{sentence}': {len(sentence_embedding)}, Time: {first_call_duration:.4f} seconds")

    # Timing subsequent calls
    start_time = time.time()
    sentence_embedding = get_sentence_embedding(sentence)
    second_call_duration = time.time() - start_time
    print(f"Second call: Embedding for '{sentence}': {len(sentence_embedding)}, Time: {second_call_duration:.4f} seconds")

    print(f"Dimension of Sentence Transformer embedding: {SENTENCE_TRANSFORMER_DIM}")

    # Test and time LazyClipModel for text
    start_time = time.time()
    clip_text_embedding = get_clip_text_embedding(sentence)
    first_clip_call_duration = time.time() - start_time
    print(f"First call: CLIP text embedding for '{sentence}': {len(clip_text_embedding)}, Time: {first_clip_call_duration:.4f} seconds")

    # Timing subsequent calls for CLIP text
    start_time = time.time()
    clip_text_embedding = get_clip_text_embedding(sentence)
    second_clip_call_duration = time.time() - start_time
    print(f"Second call: CLIP text embedding for '{sentence}': {len(clip_text_embedding)}, Time: {second_clip_call_duration:.4f} seconds")

    img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

    start_time = time.time()
    clip_image_embedding = get_image_embedding(img)
    first_img_call_duration = time.time() - start_time
    print(f"First call: CLIP image embedding: {len(clip_image_embedding)}, Time: {first_img_call_duration:.4f} seconds")

    # Timing subsequent calls for CLIP image
    start_time = time.time()
    clip_image_embedding = get_image_embedding(img)
    second_img_call_duration = time.time() - start_time
    print(f"Second call: CLIP image embedding: {len(clip_image_embedding)}, Time: {second_img_call_duration:.4f} seconds")

    print(f"Dimension of CLIP model embedding: {CLIP_DIM}")