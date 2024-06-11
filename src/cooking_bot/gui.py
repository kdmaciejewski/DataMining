import os
import textwrap

from .data_formats import *
from abc import ABC, abstractmethod
from .data_formats import QueryResult, Recipe
from typing import Optional
import tkinter as tk
from cooking_bot import REPO_PATH
from PIL import Image, ImageTk
from loguru import logger

class GuiInterface(ABC):
    @abstractmethod
    def show_single_question_and_answer_field(self, question: str) -> str:
        """Render a single question and return answer as text"""

    @abstractmethod
    def recipy_choice(self, results: QueryResult, additional_text: str) -> Optional[Recipe]:
        """
        Render the recipe recommendations and return chosen Recipe. If user wants to go back/choose another recipe return None
        """

    @abstractmethod
    def render_plan_llm_conv(self, conversation: list[tuple[str, str]], current_recipe: Recipe, step_number : int = -1) -> Optional[str]:
        """Window for during the plan LLM conversation. Show recipe and image/instruction images perhaps

        Args:
            conversation: list[tuple[str, str]] - list of tuples of whether model or user and text
            current_recipe (Recipe): _description_

        Returns:
            Optional[str]: String answer of user or None if stop/cancel

        conversation[-1][-1] is last plan LLM response
        """


class TkinterGUI(GuiInterface):
    imgsz = 150
    default_font = ("Courier", 12)
    title_font = ("Courier", 14)
    
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x400")
        self.response = tk.StringVar()
        self.not_included = Image.open(REPO_PATH/"imgs/not_included.png").resize((self.imgsz, self.imgsz))

    def show_single_question_and_answer_field(self, question: str) -> str:
        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        label = tk.Label(frame, font = self.default_font, text=question)
        label.pack(pady=10)

        entry = tk.Entry(frame, textvariable=self.response, width=50, font=self.default_font)
        entry.pack(pady=10)
        entry.bind("<Return>", lambda event: self.root.quit())


        button = tk.Button(frame, text="Submit", command=self.root.quit)
        button.pack(pady=10)

        self.root.mainloop()
        answer = self.response.get()
        self.response.set("")
        frame.destroy()
        return answer

    def recipy_choice(self, results: QueryResult, additional_text: str) -> Optional[Recipe]:
        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        label = tk.Label(frame, font = self.default_font, text=f"Here are some recipes for you: {additional_text}")
        label.pack(pady=10)

        recipe_frame = tk.Frame(frame)
        recipe_frame.pack()

        for idx, recipe in enumerate(results.hits[:5]):  
            sub_frame = tk.Frame(recipe_frame)
            sub_frame.pack(side="left", fill="both", expand=True, padx=10)

            if recipe.images and (img := recipe.images[0].get_image()) is not None:
                img = ImageTk.PhotoImage(img.resize((self.imgsz, self.imgsz)))
            else:
                img = ImageTk.PhotoImage(self.not_included.copy())


            btn = tk.Button(sub_frame, image=img, command= lambda idx = idx: self._choose_recipe(idx))
            title_label = tk.Label(sub_frame, font = self.title_font, text=recipe.displayName, wraplength=150, height=5)
            title_label.pack()
            btn.image = img 
            btn.pack()
            
            
            def text_info(recipe : Recipe) -> str:
                
                text = ""
                if recipe.totalTimeMinutes:
                    text += f"- Time: {recipe.totalTimeMinutes} minutes\n"
                
                text += f"- {len(recipe.instructions)} Steps\n"
                
                if recipe.tools:
                    text += "- Tools:\n"
                    
                    for tool in recipe.tools:
                        text += f"\t\t{tool.displayName}\n"
                        
                return text
                        

            details_label = tk.Label(sub_frame, font = self.default_font, text=text_info(recipe), wraplength=150)
            details_label.pack()

        self.chosen_index = None
        self.root.mainloop()

        frame.destroy()
        if self.chosen_index is not None:
            return results.hits[self.chosen_index]
        return None


    def _choose_recipe(self, index):
        self.chosen_index = index
        self.root.quit()

    def render_plan_llm_conv(self, conversation: list[tuple[str, str]], current_recipe: Recipe, step_number : int) -> Optional[str]:
        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        recipe_label = tk.Label(frame, font = self.title_font, text=f"Recipe: {current_recipe.displayName}")
        recipe_label.pack(pady=10)


        try: 
            images = current_recipe.instructions[step_number - 1].stepImages
            if images and (img := images[0].get_image()) is not None :
                img = ImageTk.PhotoImage(img.resize((self.imgsz * 2, self.imgsz * 2,)))
                img_label = tk.Label(frame, image = img)
                img_label.pack() 
                
        
        except Exception:
            logger.exception("Cant show image picture")


        chat_frame = tk.Frame(frame)  # Frame to contain chat display
        chat_frame.pack(pady=10)

        for talker, text in conversation[-1:]:
            wrapped_text = "\n".join(textwrap.wrap(text, width=50))  
            talk_label = tk.Label(chat_frame, font = self.default_font, text=f"{wrapped_text}", justify=tk.LEFT)
            talk_label.pack(anchor="w") 
            tk.Label(chat_frame, font = self.default_font, text="").pack()

        entry = tk.Entry(frame, textvariable=self.response, width=50, font=self.default_font)
        entry.pack(pady=10)
        entry.bind("<Return>", lambda event: self.root.quit())

        button = tk.Button(frame, text="Submit", command=self.root.quit)
        button.pack(pady=10)

        self.root.mainloop()
        user_response = self.response.get()
        self.response.set("")
        frame.destroy()
        return user_response if user_response.lower() != 'stop' else None



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
        
    def render_plan_llm_conv(self, conversation : list[tuple[str,str]], current_recipe: Recipe, step_number : int = -1) ->  Optional[str]:
        self.clear()
        print(f"Recipe: {current_recipe.displayName}")
        
        # if current_recipe.images and (img := current_recipe.images[0].get_image()):
        #     img.show()
        
        for talker, text in conversation[-1:]:
            
            print(talker, " : ",text, "\n")
            
        return input("\nYou:\n")
        
        
    
    
