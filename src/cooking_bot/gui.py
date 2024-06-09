import os
import textwrap

from .data_formats import *
from abc import ABC, abstractmethod
from .data_formats import QueryResult, Recipe
from typing import Optional
import tkinter as tk
from PIL import Image, ImageTk


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
    def render_plan_llm_conv(self, conversation: list[tuple[str, str]], current_recipe: Recipe) -> Optional[str]:
        """Window for during the plan LLM conversation. Show recipe and image/instruction images perhaps

        Args:
            conversation: list[tuple[str, str]] - list of tuples of whether model or user and text
            current_recipe (Recipe): _description_

        Returns:
            Optional[str]: String answer of user or None if stop/cancel

        conversation[-1][-1] is last plan LLM response
        """


class TkinterGUI(GuiInterface):
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("400x415")
        self.response = tk.StringVar()

    def show_single_question_and_answer_field(self, question: str) -> str:
        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        label = tk.Label(frame, text=question)
        label.pack(pady=10)

        entry = tk.Entry(frame, textvariable=self.response, width=50)  # Adjusting width of entry field
        entry.pack(pady=10)

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

        label = tk.Label(frame, text=f"Here are some recipes for you: {additional_text}")
        label.pack(pady=10)

        images = [ImageTk.PhotoImage(recipe.images[0].get_image().resize((100, 100))) for recipe in results.hits[:3]]
        buttons = []

        for idx, (img, recipe) in enumerate(zip(images, results.hits[:3])):
            btn_frame = tk.Frame(frame)
            btn_frame.pack()

            btn = tk.Button(btn_frame, image=img, command=lambda idx=idx: self._choose_recipe(idx))
            btn.image = img  # Keep a reference to avoid garbage collection
            btn.pack(side="left", padx=5)
            buttons.append(btn)

            title_label = tk.Label(btn_frame, text=recipe.displayName, wraplength=150, width=20)  # Adjust wrap length as needed
            title_label.pack(side="left", padx=5)  # Adding title label below the image button

        self.chosen_index = None
        self.root.mainloop()

        frame.destroy()
        if self.chosen_index is not None:
            return results.hits[self.chosen_index]
        return None

    def _choose_recipe(self, index):
        self.chosen_index = index
        self.root.quit()

    def render_plan_llm_conv(self, conversation: list[tuple[str, str]], current_recipe: Recipe) -> Optional[str]:
        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        recipe_label = tk.Label(frame, text=f"Recipe: {current_recipe.displayName}")
        recipe_label.pack(pady=10)

        chat_frame = tk.Frame(frame)  # Frame to contain chat display
        chat_frame.pack(pady=10)

        for talker, text in conversation[-1:]:
            wrapped_text = "\n".join(textwrap.wrap(text, width=50))  # Word wrapping for model responses
            talk_label = tk.Label(chat_frame, text=f"{talker}: {wrapped_text}", justify=tk.LEFT)  # Adjusting text alignment
            talk_label.pack(anchor="w")  # Anchoring labels to the left
            tk.Label(chat_frame, text="").pack()  # Adding space between chat entries

        entry = tk.Entry(frame, textvariable=self.response, width=50)  # Adjusting width of entry field
        entry.pack(pady=10)

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
        
    def render_plan_llm_conv(self, conversation : list[tuple[str,str]], current_recipe: Recipe) ->  Optional[str]:
        self.clear()
        print(f"Recipe: {current_recipe.displayName}")
        
        # if current_recipe.images and (img := current_recipe.images[0].get_image()):
        #     img.show()
        
        for talker, text in conversation[-1:]:
            
            print(talker, " : ",text, "\n")
            
        return input("\nYou:\n")
        
        
    
    
