from cooking_bot import REPO_PATH
from .plan_llm import test_ping, send_message, PromptSettings
from .manager_utils import Intents, Answer, get_awnser, get_intent, get_best_answer, RecipyCategory, get_recipy_category
from .data_formats import *
from .querries import (
    get_recipy_by_ingredients,
    query_recipies,
    ingredient_similarity_search,
    get_most_similar_step,
)
import time
from loguru import logger
from enum import Enum
from nltk.corpus import stopwords
import random

stop_words = set(stopwords.words('english'))

PLAN_INTENTS = {
    Intents.GoToStepIntent,
    Intents.ChitChatIntent,
    Intents.NextStepIntent,
    Intents.MoreDetailIntent,
    Intents.CompleteTaskIntent,
    Intents.RepeatIntent,
    Intents.IngredientsConfirmationIntent,
    Intents.StartStepsIntent,
}





class States(Enum):
    RECIPE_CHOISE = 0
    PLAN_LLM_CONV = 1
    
    
class RecipeChoiceSubstates(Enum):
    Find = 0
    AWAITING_APROVAL = 1
    
class RecipeChoice:
    
    n_suggestions = 5
    name = States.RECIPE_CHOISE
    
    def __init__(self) -> None:
        self.state = RecipeChoiceSubstates.Find
        self.suggestions : QueryResult = None 
        
    
    def dosth(self, message : str, intent : Intents) -> tuple[str, States]:
        
        
        
        if self.state == RecipeChoiceSubstates.Find:
            return self._find_state(message, intent), States.RECIPE_CHOISE
        
        elif self.state == RecipeChoiceSubstates.AWAITING_APROVAL:
            return self._waiting_state(message, intent)
        
        else:
            raise ValueError("Wrong state")
            
    def _find_state(self, message : str, intent : Intents) -> str:
        
        if intent not in [Intents.SelectIntent, Intents.SuggestionsIntent, Intents.IdentifyProcessIntent]:
            logger.debug(f"Intent not for recipy decision {intent}")
            return "I am sorry. I can't understand you. What would you like to cook today?"
        
        start = time.time()
        embedding = get_sentence_embedding(message)
        logger.debug(f"sentence emb took {time.time() - start}")
        
        category = get_recipy_category(embedding)
        logger.debug(f"Detected category {category}")
        
        response, succes = self.query_recipe(message, category)
        if succes:
            self.state = RecipeChoiceSubstates.AWAITING_APROVAL
            
            
        return response
    def _waiting_state(self, message : str, intent : Intents)-> tuple[str, States]:
        
        if intent in [Intents.FallbackIntent, Intents.PreviousStepIntent, Intents.StartStepsIntent, Intents.FallbackIntent, Intents.NoIntent]:
            self.state = RecipeChoiceSubstates.Find
            self.suggestions = None
            return "I am sorry that you don't like recipy. Let's try again. What would you like to cook today?", States.RECIPE_CHOISE
        
        if intent in [Intents.NextStepIntent, Intents.YesIntent]:
            
            return "", States.PLAN_LLM_CONV
            
        logger.debug(f"Cant undestand intent in waiting {intent}")
        return "Sorry, I struggle to understand that. Would you like us to cook choose the recipy?", States.RECIPE_CHOISE
        
    
    def _render_suggestion(self, suggestion : QueryResult):
        
        
        s = f"""How about this recipe?\n\t{suggestion.hits[0].displayName}"""
        self.final_suggestion = suggestion.hits[0]
        return s
        
        
    def query_recipe(self, message, category):
        
        if category == RecipyCategory.General:
            
            res = query_recipies(message, mode="clip", size=self.n_suggestions)
            if res.n_hits == 0:
                logger.debug("Clip did not work. Fall back to random choice")
                res = query_recipies(size=self.n_suggestions * 6)
                
                res.n_hits = min(self.n_suggestions, res.n_hits)
                res.hits = random.choices(res.hits, k = min(self.n_suggestions, res.n_hits))
            
            self.suggestions = res
                    
            
        elif category == RecipyCategory.Specific:
            
            res = query_recipies(message, mode="text", size=self.n_suggestions)
            if res.n_hits == 0:
            
                logger.debug("text did not work. use vec")
                res = query_recipies(message, mode="vec", size=self.n_suggestions, min_cos_sim=-1)
            
            self.suggestions = res
            
        
        elif category == RecipyCategory.Ingredient:
            
            ingredients = get_awnser(message, "What ingredients did the user list?").text
            ingredients = ingredients.lower().replace(",","").split(" ")
            ingredients = [i for i in ingredients if i not in stop_words]
            
            logger.debug(f"Extrated ingredients {ingredients}")
            if not ingredients:
                return "I could sadly not understand your ingredients", False
            
            
            res = get_recipy_by_ingredients(ingredients)
            if res.n_hits == 0:
                return "Sadly I dont have a matching recipe", False
            self.suggestions = res
            
            
        return self._render_suggestion(self.suggestions), True
            
            
            
plan_llm_settings = PromptSettings()
plan_llm_timeout = 10

class PlanLLMConv:
    name = States.PLAN_LLM_CONV
    
    
    
    INIT_TEMPLATE = "<|prompter|>  You are a taskbot tasked with helping users cook recipes or DIY projects. \
            I will give you a recipe and I want you to help me do it step by step. You should always be \
            empathetic, honest, and should always help me. If I ask you something that does not relate\
            to the recipe you should politely reject the request and try too get me focused on the recipe.\
            I am unsure how to cook something or do something related to the recipe you should help me to \
            the best of your ability. Please use a {} tone of voice. <|endofturn|> <|prompter|>\
            The Recipy is called {} and the instruction list is: {}.\
            The converstaion is already flowing, so start guiding the user in cooking {} with by the first step {}.<|endofturn|> <|assistant|> "
    def _get_init_promt(self, recipe: Recipe):
        recipe_name = recipe.displayName
        system_tone = "neutral"
        instruction_string = "\n".join(
            [f"Step {ins.stepNumber}: {ins.stepText}" for ins in recipe.instructions]
        )
        return self.INIT_TEMPLATE.format(
            system_tone,
            recipe_name,
            instruction_string,
            recipe_name,
            recipe.instructions[0].stepText,
        )
    
    def call_api(self, user_input = None):
        
        if user_input is not None:
            self.history +=  "<|prompter|> " + user_input + " <|endofturn|> <|assistant|> "
        
        model_response = send_message(self.history, plan_llm_settings, plan_llm_timeout)
        
        if model_response is None:
            return "Sorry, we are having some techical Difficulties."
            
        self.history += model_response + " <|endofturn|>"    
        return model_response
    
    def dosth(self, message : str, intent : Intents) -> tuple[str, States]:
        
        return self.call_api(message), States.PLAN_LLM_CONV
    
    
    def __init__(self, recipe : Recipe):    
        
        self.history = self._get_init_promt(recipe)
        

class DialogManager:


    def __init__(self) -> None:

        
        self.recipe : Optional[Recipe] = None
        
        self.state = RecipeChoice()

    
        
    
    def compute_answer(self, message : str = None) -> Optional[str]:
        
        if message is None:
            message = input("\nYou:\n")
        
        intent = get_intent(message)
        logger.debug(f"Message '{message}' - {intent}")
        
        if intent == Intents.StopInten:
            
            if self.state.name  == States.PLAN_LLM_CONV:
                
                self.state = RecipeChoice()
                return "Do you want to try a different Recipe?"
                
            logger.info("Stopping conversation")
            return None
        elif intent == Intents.InappropriateIntent:
            return "Please, this is inapporiate."

        response, next_state = self.state.dosth(message, intent)
        
        if self.state.name == States.RECIPE_CHOISE and next_state == States.PLAN_LLM_CONV:
            logger.debug("Switching to Plan llm")
            self.state = PlanLLMConv(self.state.final_suggestion)
            response = self.state.call_api()
        
        print(response)
        
        return ""
    

def main():
    
    test_ping(timeout=plan_llm_timeout)
    get_sentence_embedding("load model")
    manager = DialogManager()  
    
    print("Hey, what would you like to cook today?")      
    while manager.compute_answer() is not None:
        pass
        
    
    


