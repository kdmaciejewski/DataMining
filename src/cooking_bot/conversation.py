import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .plan_llm import test_ping, send_message, PromptSettings
from .intent_detector import IntentDetector
from .intents import Intents, RecipyIntent
from cooking_bot.recipy_query import get_recipes
from .data_formats import *
from loguru import logger
from .gui import GuiInterface, CLI_GUI


intent_detector = IntentDetector()
GUI : GuiInterface = CLI_GUI()

    
    
class RecipeChoice:
    
    n_suggestions = 5    
        
    def recipe_search(self, question : str = "Hey, what would you like to cook today?") -> Optional[Recipe]:
        
        res = GUI.show_single_question_and_answer_field(question)
        
        intent : Intents = intent_detector.get_intent(s = res, agent_prompt= question)
        
        if intent == Intents.StopIntent:
            return
        
        return self._choose_recipy(res, intent=intent)
                

    def _choose_recipy(self, message : str, intent : Intents):
        
        if intent not in [Intents.SelectIntent, Intents.SuggestionsIntent, Intents.IdentifyProcessIntent]:
            logger.debug(f"Intent not for recipy decision {intent}")
            return self.recipe_search("I am sorry. I can't understand you. What would you like to cook today?")
        
        suggestions, add_string = self.query_recipe(message=message)
        
        if suggestions.n_hits == 0:
            
            return self.recipe_search(f"Sadly I dont have a matching recipe{add_string}. Let's try again. What would you like to cook?")
        
        
        chosen_recipe = GUI.recipy_choice(suggestions, add_string)
        
        if chosen_recipe is None:
            return self.recipe_search(f"Let's try to find you something else. What would you like to cook?")
            
        return chosen_recipe
        
    def query_recipe(self, message : str) -> tuple[QueryResult, str]:
        
        intent : RecipyIntent = intent_detector.get_recipy_intent(message)
        
        suggestion, max_minutes, max_steps = get_recipes(message = message, intent= intent, n_suggestions= self.n_suggestions)
        
        add_string = ""
        
        if max_minutes is not None:
            add_string = f", which take max {max_minutes} minutes"
        if max_steps and add_string != "":
            add_string += " and have few steps" if add_string else ",which have few steps"

        return suggestion, add_string
            
        

plan_llm_settings = PromptSettings()
plan_llm_timeout = 10

class PlanLLMConversation:
    
    INIT_TEMPLATE = '<|prompter|>  You are a taskbot tasked with helping users cook recipes or DIY projects. \
            I will give you a recipe and I want you to help me do it step by step. You should always be \
            empathetic, honest, and should always help me. If I ask you something that does not relate\
            to the recipe you should politely reject the request and try too get me focused on the recipe.\
            I am unsure how to cook something or do something related to the recipe you should help me to \
            the best of your ability. Please use a {} tone of voice. \
            The Recipy is called {} and the instruction list is: {}. <|endofturn|> <|prompter|>\
            Start of with a sentence like: "Lets start of with the recipy by doing {}.<|endofturn|> <|assistant|> '
            
    def __init__(self, recipe : Recipe):
            
        self.history = self._get_init_promt(recipe)
            
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
            recipe.instructions[0].stepText,
        )
    
    def get_response(self, user_input = None):
        
        if user_input is not None:
            self.history +=  "<|prompter|> " + user_input + " <|endofturn|> <|assistant|> "
        
        model_response = send_message(self.history, plan_llm_settings, plan_llm_timeout)
        
        if model_response is None:
            raise ConnectionError("Can't reach Plan LLM API")
            
        self.history += model_response + " <|endofturn|>"    
        return model_response
        

        
        
def dialog(init_message : Optional[str] = None):
    
    recipe = RecipeChoice().recipe_search() if init_message is None else RecipeChoice().recipe_search(init_message)
    
    if recipe is None:
        
        logger.debug("Recipy is None -> Stop Converstation")
        return
    
    plan_llm_conv = PlanLLMConversation(recipe=recipe)
    
    conversation : list[tuple[str,str]] = [("model", plan_llm_conv.get_response())]
    
    while (response := GUI.render_plan_llm_conv(conversation=conversation, current_recipe=recipe)) is not None:
        
        intent = intent_detector.get_intent(s = response, agent_prompt=conversation[-1][1])
        
        if intent == Intents.StopIntent:
            break
        
        conversation.append(("user", response))
        
        conversation.append(("model",plan_llm_conv.get_response( response)))
    
    dialog("Okay. Would you like to cook another recipe? If not say stop - else tell me what you would like to cook?")
    
        
        
def main():
    
    test_ping(timeout=plan_llm_timeout)
    get_sentence_embedding("load model")
    
    dialog()    
    
    


