import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from cooking_bot.plan_llm import test_ping, send_message, PromptSettings
from cooking_bot.intent_detector import IntentDetector
from cooking_bot.intents import Intents, RecipyIntent
from cooking_bot.manager_utils import get_step_numbers
from cooking_bot.recipy_query import get_recipes
from cooking_bot.data_formats import *
from loguru import logger
from cooking_bot.gui import GuiInterface, CLI_GUI, TkinterGUI

intent_detector = IntentDetector()
GUI : GuiInterface = TkinterGUI()
N_SUGGESTIONS = 5


def query_recipe(message : str) -> tuple[QueryResult, str]:
    
    intent : RecipyIntent = intent_detector.get_recipy_intent(message)
    
    suggestion, max_minutes, max_steps = get_recipes(message = message, intent= intent, n_suggestions= N_SUGGESTIONS)
    
    add_string = ""
    
    if max_minutes is not None:
        add_string = f", which take max {max_minutes} minutes"
    if max_steps and add_string != "":
        add_string += " and have few steps" if add_string else ",which have few steps"

    return suggestion, add_string
    
    
def recipe_search(init_question) -> Optional[QueryResult]:
    
    question = init_question
    
    while True:
        
        message = GUI.show_single_question_and_answer_field(question)
        intent : Intents = intent_detector.get_intent(s = message, agent_prompt= question)
        
        if intent == Intents.StopIntent:
            return
        
        if intent not in [Intents.SelectIntent, Intents.SuggestionsIntent, Intents.IdentifyProcessIntent]:
            logger.debug(f"Intent not for recipy decision {intent}")
            question = "I am sorry. I can't understand you. What would you like to cook today?"
            continue
        
        suggestions, add_string = query_recipe(message=message)
        
        if suggestions.n_hits == 0:
            
            question = f"Sadly I dont have a matching recipe{add_string}. Let's try again. What would you like to cook?"
            continue
        
        
        chosen_recipe = GUI.recipy_choice(suggestions, add_string)
        
        if chosen_recipe is None:
            question = f"Let's try to find you something else. What would you like to cook?"
            continue
            
        return chosen_recipe
        

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
        self.step_number = 1
            
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
            
        number = get_step_numbers(model_response)
        if number is not None:
            self.step_number = number
            
        self.history += model_response + " <|endofturn|>"    
        return model_response
        
        
def dialog(init_message : str = "Hey, what would you like to cook today?"):
    
    message = init_message
    
    while True:
        
        recipe = recipe_search(init_question= message)
        
        if recipe is None:
            
            logger.debug("Recipy is None -> Stop Converstation")
            return
        
        plan_llm_conv = PlanLLMConversation(recipe=recipe)
        
        conversation : list[tuple[str,str]] = [("model", plan_llm_conv.get_response())]
        
        while (response := GUI.render_plan_llm_conv(conversation=conversation, current_recipe=recipe, step_number = plan_llm_conv.step_number)) is not None:
            
            intent = intent_detector.get_intent(s = response, agent_prompt=conversation[-1][1])
            
            if intent == Intents.StopIntent:
                break
            
            conversation.append(("user", response))
            
            conversation.append(("model",plan_llm_conv.get_response( response)))
        
        message = "Okay. Would you like to cook another recipe? If not say stop - else tell me what you would like to cook?"        
            
        
def main():
    
    test_ping(timeout=plan_llm_timeout)
    get_sentence_embedding("load model")
    
    dialog()
    GUI.run()
    


if __name__ == "__main__":
    main()