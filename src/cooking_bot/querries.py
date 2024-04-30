from typing import Optional
import numpy as np
from .data_formats import *
from .client import CLIENT, INDEX_NAME



def l2(vec):
    return np.sqrt(np.sum(np.square(vec)))


def cosine_sim(a, b):

    return np.dot(a, b) / (l2(a) * l2(b))


def query_recipies(
    name: Optional[str] = None,
    mode: str = "vec",
    size: int = 5,
    k: int = 2,
    enforce_images: bool = False,
    max_total_minutes: Optional[int] = None,
    max_steps: Optional[int] = None,
    min_cos_sim: float = 0.0,
) -> QueryResult:
    """
    Query recipes from the OpenSearch index based on various criteria.

    Parameters:
    - name (str, optional): Name or part of the name of the recipe.
    - mode (str, optional): The mode of querying, 'vec' for vector search or 'text' for text search.
    - size (int, optional): The number of search results to return.
    - k (int, optional): The number of nearest neighbors to retrieve (for vector search).
    - enforce_images (bool, optional): Whether to only include recipes with images.
    - max_total_minutes (int, optional): Maximum total preparation time of the recipe.
    - max_steps (int, optional): Maximum number of steps in the recipe.
    - min_cos_sim (float, optional): Minimum cosine similarity for vector search results.

    Returns:
    - QueryResult: The search results meeting the query criteria.
    """

    if mode not in ["text", "vec"]:

        raise ValueError("Wrong mode")

    post_query_filters: List[callable] = []

    inner = {"bool": {"must": []}}

    if mode == "vec" and name:

        embedding = get_sentence_embedding(name)

        inner["bool"]["must"].append(
            {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": k,
                    }
                }
            }
        )

        def filter_min_sim(hit: QueryHit) -> bool:

            return cosine_sim(hit.recipe.embedding, embedding) >= min_cos_sim

        post_query_filters.append(filter_min_sim)

    elif mode == "text" and name:
        inner["bool"]["must"].append(
            {
                "multi_match": {
                    "query": name,
                    "fields": ["displayName", "description"],
                }
            }
        )

    if enforce_images:

        inner["bool"]["must"].append(
            {
                "nested": {
                    "path": "images",
                    "query": {"bool": {"must": [{"exists": {"field": "images.url"}}]}},
                }
            },
        )

    if max_total_minutes is not None:
        inner["bool"]["must"].append(
            {"range": {"totalTimeMinutes": {"lte": max_total_minutes}}}
        )

    query = {
        "size": size,
        "query": inner,
    }

    response = create_QueryResult(
        CLIENT.search(body=query, index=INDEX_NAME), size=size
    )

    if max_steps is not None:

        response.hits = [
            i for i in response.hits if len(i.recipe.instructions) <= max_steps
        ]

    if post_query_filters:

        response.hits = [
            i for i in response.hits if all(func(i) for func in post_query_filters)
        ]

    return response


def get_recipy_by_ingredients(
    ingredients: List[str],
    min_should: int = -1,
    size: int = 5,
    enforce_images: bool = False,
    max_total_minutes: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> QueryResult:
    """
    Retrieve recipes based on a list of ingredients.

    Parameters:
    - ingredients (List[str]): List of ingredients to search for.
    - min_should (int, optional): Minimum number of ingredients that should match.
    - size (int, optional): The number of search results to return.
    - enforce_images (bool, optional): Whether to only include recipes with images.
    - max_total_minutes (int, optional): Maximum total preparation time of the recipe.
    - max_steps (int, optional): Maximum number of steps in the recipe.

    Returns:
    - QueryResult: The search results meeting the query criteria.
    """

    inner = {
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": "ingredients",
                        "query": {
                            "multi_match": {
                                "query": ingredient,
                                "fields": [
                                    "ingredients.displayText",
                                    "ingredients.ingredient",
                                ],
                            }
                        },
                    }
                }
                for ingredient in ingredients
            ],
            "minimum_should_match": (
                min_should if min_should > 0 else len(ingredients)
            ),
            "must": [],
        }
    }

    if enforce_images:

        inner["bool"]["must"].append(
            {
                "nested": {
                    "path": "images",
                    "query": {"bool": {"must": [{"exists": {"field": "images.url"}}]}},
                }
            },
        )

    if max_total_minutes is not None:
        inner["bool"]["must"].append(
            {"range": {"totalTimeMinutes": {"lte": max_total_minutes}}}
        )

    query = {"size": size, "query": inner}

    response = create_QueryResult(CLIENT.search(body=query, index=INDEX_NAME), size)

    if max_steps is not None:

        response.hits = [
            i for i in response.hits if len(i.recipe.instructions) <= max_steps
        ]
    return response


def ingredient_similarity_search(ingredient: str, size=5) -> List[Ingredient]:
    """
    Perform an ingredient similarity search to find matching or related ingredients.
    
    Parameters:
    - ingredient (str): The ingredient to search for.
    - size (int, optional): The number of search results to return.
    
    Returns:
    - List[Ingredient]: A list of ingredients sorted by similarity to the queried ingredient.
    """
    
    ingredient_embedding = get_sentence_embedding(ingredient)

    query = {
        "size": size,
        "query": {
            "nested": {
                "path": "ingredients",
                "query": {
                    "knn": {
                        "ingredients.embedding": {
                            "vector": ingredient_embedding,
                            "k": 2,
                        }
                    }
                },
            }
        },
    }

    resonse = create_QueryResult(CLIENT.search(body=query, index=INDEX_NAME), size)

    ingredients = []
    for hit in resonse.hits:
        sorted_ingredienst = sorted(
            hit.recipe.ingredients,
            key=lambda x: cosine_sim(x.embedding, ingredient_embedding),
            reverse=True,
        )

        if sorted_ingredienst:
            ingredients.append(sorted_ingredienst[0])

    return ingredients
