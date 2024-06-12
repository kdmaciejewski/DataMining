from cooking_bot.manager_utils import get_awnser
from cooking_bot.intents import TimeCategory, Difficulty, RecipyCategory, RecipyIntent
from cooking_bot.data_formats import *
from cooking_bot.queries import (
    get_recipy_by_ingredients,
    query_recipies,
)
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import re
import random

stop_words = set(stopwords.words('english'))



DEFAULT_SHORT = 25
EASY_MAX_STEPS = 7

def calc_max_minutes(message : str, time : TimeCategory):
    
    if time == TimeCategory.Urgent:
        return DEFAULT_SHORT
    
    elif time == TimeCategory.No:
        return None
    
    tokens = wordpunct_tokenize(message)
    
    
    if "minutes" in tokens or "mins" in tokens:
        
        answer = get_awnser(message, "How many minutes should the recipy take maximum?")
        
        numbers = list(map(int, re.findall(r"\d+",answer.text)))
        
        if len(numbers) == 0:
            logger.debug(f"Cant find minutes in {answer.text}")
            return None
            
        minutes = max(numbers)
        
        logger.debug(f"Choose max {minutes} from {answer}")
        return minutes
    
    elif "hour" in tokens or "hours" in tokens:
        
        answer = get_awnser(message, "How many hours should the recipy take maximum?")
        
        numbers = list(map(int, re.findall(r"\d+",answer.text)))
        
        if len(numbers) == 0:
            logger.debug(f"Cant find minutes in {answer.text}")
            return None
            
        minutes = max(numbers) * 60
        
        logger.debug(f"Choose max {minutes} from {answer}")
        return minutes
    
    else:
        
        numbers = list(map(int, re.findall(r"\d+", message)))
        
        if len(numbers) == 0:
            logger.debug("Can't find specific time")
            return
        
        logger.debug(f"Did not detect time scale, defaulting to {numbers[0]} minutes")
        
        return numbers[0]
            
            


def get_max_steps(intent : Difficulty):
    
    if intent == Difficulty.Easy: return EASY_MAX_STEPS
    
    return None



def get_recipes(message : str, intent : RecipyIntent, n_suggestions : int) -> tuple[QueryResult, int, int]:
    
    max_steps = get_max_steps(intent.diff)
    max_minutes = calc_max_minutes(message, intent.time)
    
    res : QueryResult
    
    if intent.category == RecipyCategory.LookBased:
        
        res = query_recipies(message, mode = "clip", size = n_suggestions, min_cos_sim= -1, max_steps= max_steps, max_total_minutes = max_minutes)
    
    
    elif intent.category == RecipyCategory.Specific:
        
        name =  get_awnser(message, "What is the name of the specific recipy?").text
        logger.debug(f"Extracted name - '{name}'")
        
        res = query_recipies(name, mode = "text", size = n_suggestions, max_steps= max_steps, max_total_minutes = max_minutes)
    
    elif intent.category == RecipyCategory.General:
            
        res = query_recipies(message, mode = "clip", size = n_suggestions, min_cos_sim= .2, max_steps= max_steps, max_total_minutes = max_minutes)
        
        if res.n_hits == 0:
            logger.debug("Clip did not work. Fall back to random choice")
            
            res = query_recipies(size = n_suggestions * 20, max_steps= max_steps, max_total_minutes = max_minutes)
            
            res.n_hits = min(n_suggestions, res.n_hits)
            res.hits = random.choices(res.hits, k = res.n_hits)

    elif intent.category == RecipyCategory.Ingredient:
            
        ingredients = get_awnser(message, "What ingredients did the user list?").text
        ingredients = ingredients.lower().replace(",","").split(" ")
        ingredients = [i for i in ingredients if i not in stop_words]
        
        logger.debug(f"Extrated ingredients {ingredients}")
        if not ingredients:
            return "I could sadly not understand your ingredients", False
        
        
        res = get_recipy_by_ingredients(ingredients, max_total_minutes=max_minutes, max_steps=max_steps, size=n_suggestions)
        
        
    return res, max_minutes, max_steps

    
if __name__ == "__main__":
    
    examples = [
        ("chicken sandwhich in under 10", RecipyIntent(category=0, time=1, diff=0)),
        ("Something that looks healthy", RecipyIntent(category=0, time=0, diff=0)),
        ("Something that looks healthy and is quick", RecipyIntent(category=0, time=2, diff=0)),
        ("Something that looks healthy and needs to be quick", RecipyIntent(category=0, time=1, diff=0)),
        ("Something that looks healthy and takes max 20 minutes", RecipyIntent(category=0, time=1, diff=0)),
        ("Something that looks healthy and takes max 1 hour", RecipyIntent(category=0, time=1, diff=0)),
        ("I want to cook spaghetti bolognese with an easy recipe", RecipyIntent(category=2, time=0, diff=1)),
        ("Show me how to make a quick chicken curry", RecipyIntent(category=2, time=2, diff=0)),
        ("Can I have a recipe for a quick beef stew?", RecipyIntent(category=2, time=2, diff=0)),
        ("I need a simple recipe for pancakes", RecipyIntent(category=2, time=0, diff=1)),
        ("What's a quick dessert I can whip up in 30 minutes?", RecipyIntent(category=0, time=1, diff=0)),
        ("Give me a challenging recipe for a seafood dish", RecipyIntent(category=2, time=0, diff=0)),
        ("What can I cook for dinner thats simple and quick?", RecipyIntent(category=3, time=2, diff=1)),
        ("Need a recipe with eggs, rice and onions that is easy", RecipyIntent(category=1, time=0, diff=1)),
    ]
    
    for message, intent in examples:
        print(f"Query: {message} - Intent: {intent.category}, {intent.time}, {intent.diff}")
        
        results, max_minutes, max_steps = get_recipes(message, intent, n_suggestions=1)
        print(f"Max Minutes: {max_minutes}, Max Steps: {max_steps}")
        
        if len(results.hits) == 0:
            print("No match")
            continue
        
        hit = results.hits[0]
        print(f"Recipe Suggested: {hit.displayName} - {hit.totalTimeMinutes} - {len(hit.instructions)}\n\n","-"*10)

