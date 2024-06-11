from optimum.pipelines import pipeline
from cooking_bot import REPO_PATH
from pydantic import BaseModel
from typing import List, Union
import re


class Answer(BaseModel):
    score: float
    text: str



qpipe = pipeline(
    "question-answering",
    model=str(REPO_PATH / "models/roberta-base-squad2"),
    tokenizer="deepset/roberta-base-squad2",
    accelerator="ort",
)


def get_awnser(context: str, question: str) -> Answer:

    prompt = {"context": context, "question": question}
    res = qpipe(prompt)
    return Answer(score=res["score"], text=res["answer"])


def get_best_answer(contexts :Union[ List[str] , str], questions:Union[ List[str] , str]) -> tuple[Answer, int]:
    
    awnsers = []
    
    if isinstance(contexts, str):
        contexts = [contexts]
    if isinstance(questions, str):
        questions = [questions]
        
    for con in contexts:
        for q in questions:
            awnsers.append(get_awnser(con, q))

    best_id = max(range(len(awnsers)), key= lambda x: awnsers[x].score)
    return awnsers[best_id], best_id


def get_step_numbers(text):
    # Regex to match the pattern
    pattern = r"(?i)\bstep:?\s*(\d+)"
    
    # Search the text for the first occurrence of the pattern
    match = re.search(pattern, text)
    
    if match:
        # Return the first captured group, which is the digits
        return int(match.group(1))
    else:
        # Return None if no match is found
        return None