import json as json
from tqdm import tqdm
from data_formats import *
import json

with open("recipes_data.json", "r") as read_file:
    data = json.load(read_file)


as_pydantic = [create_Recipy(d) for d in tqdm(data.values(), desc="Computing Embeddings")]

as_json = [i.model_dump() for i in as_pydantic]

with open("data_with_embeddings.json","w") as f:
    json.dump(as_json, f)