from .data_formats import *
from abc import ABC, abstractmethod
from .data_formats import QueryResult, Recipe
from typing import Optional
import os

class GuiInterface(ABC):
    
    @abstractmethod
    def show_single_question_and_answer_field(self, question : str) -> str:
        """Render a single question and return answer as text"""
        
    @abstractmethod
    def recipy_choice(self, results : QueryResult, additional_text :str) -> Optional[Recipe]:
        """
        Render the recipy recommendations and return chosen Recipe. If user wants to go back/choose another recipy return None
        """
    
    
    def render_plan_llm_conv(self, conversation : list[tuple[str,str]], current_recipe : Recipe) -> Optional[str]:
        """Window for during the plan llm conversation. Show recipy and image/ instruction images perhabs

        Args:
            conversation : list[tuple[str,str]] - list of tuples of wether model or user and text
            current_recipe (Recipe): _description_

        Returns:
            Optional[str]: String answer of user or None if stop/cancel
            
            
        conversation[-1][-1] is last plan llm resonse
        """


class CLI_GUI(GuiInterface):
    
    
    def clear(self):
        if os.name == 'posix':
            
            os.system("clear")
        else: 
            os.system("cls")
        
    def show_single_question_and_answer_field(self, question : str) -> str:
        
        self.clear()
        print(question)
        
        return input("\nYou:\n")
    
    
    def recipy_choice(self, results: QueryResult, additional_text: str) ->  Optional[Recipe]:
        
        self.clear()
        print("We have this recipy for you: \n\t" + results.hits[0].displayName+ additional_text)
        
        if input("do you like it? (return y)") == "y":
            return results.hits[0]
        
        else:
            return None
        
    def render_plan_llm_conv(self, conversation : list[tuple[str,str]], current_recipe: Recipe) ->  Optional[str]:
        self.clear()
        print(f"Recipe: {current_recipe.displayName}")
        
        # if current_recipe.images and (img := current_recipe.images[0].get_image()):
        #     img.show()
        
        for talker, text in conversation[-1:]:
            
            print(talker, " : ",text, "\n")
            
        return input("\nYou:\n")
        
        
    
    
