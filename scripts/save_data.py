import json as json
from tqdm import tqdm
from cooking_bot.data_formats import *
import json

with open("recipes_data.json", "r") as read_file:
    data = json.load(read_file)


failed = []

for name, d in tqdm(data.items(), desc="Computing Embeddings"):

    try:
        d = create_Recipy(d, calc_embeddings=True)
        
        with open(f"jsons/{name}.json", "w") as f:
            json.dump(d.model_dump(), f)

    except Exception as e:
        print("Failed ", e)
        failed.append(name)

    
with open("failed.json","w") as f:
    json.dump(failed, f)
    
