from optimum.pipelines import pipeline
from cooking_bot import REPO_PATH
from pydantic import BaseModel
from typing import List, Union



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