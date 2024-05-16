from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from optimum.pipelines import pipeline
import numpy as np
from cooking_bot import REPO_PATH
from enum import Enum
from pydantic import BaseModel
from typing import List, Union
from sklearn.svm import SVC
import pickle

class Intents(Enum):

    GetCuriositiesIntent = 0
    GreetingIntent = 1
    SelectIntent = 2
    ShowStepsIntent = 3
    IdentifyRestrictionsIntent = 4
    ProvideUserNameIntent = 5
    MoreOptionsIntent = 6
    RepeatIntent = 7
    HelpIntent = 8
    QuestionIntent = 9
    MoreDetailIntent = 10
    AdjustServingsIntent = 11
    GoToStepIntent = 12
    SetTimerIntent = 13
    OutOfScopeIntent = 14
    FallbackIntent = 15
    PreviousStepIntent = 16
    TerminateCurrentTaskIntent = 17
    ChitChatIntent = 18
    CompleteTaskIntent = 19
    NoneOfTheseIntent = 20
    ShoppingIntent = 21
    PauseIntent = 22
    CancelIntent = 23
    StartStepsIntent = 24
    InappropriateIntent = 25
    NoIntent = 26
    SuggestionsIntent = 27
    ResumeTaskIntent = 28
    IngredientsConfirmationIntent = 29
    NextStepIntent = 30
    IdentifyProcessIntent = 31
    NoRestrictionsIntent = 32
    YesIntent = 33
    SubstitutionIntent = 34
    StopInten = 35


class Answer(BaseModel):
    score: float
    text: str


twiz_model = ORTModelForSequenceClassification.from_pretrained(REPO_PATH / "models/twiz-intent")
twiz_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

qpipe = pipeline(
    "question-answering",
    model=str(REPO_PATH / "models/roberta-base-squad2"),
    tokenizer="deepset/roberta-base-squad2",
    accelerator="ort",
)

with open(str(REPO_PATH / "models/recipe_query_svm"), "rb") as f:
    recipy_query_svm : SVC = pickle.load(f)

def get_intent(
    s: str,
    agent_prompt: str = "I can help you finding delicious recipes. What kind of recipe would you like to search for?",
) -> Intents:
    tokens = twiz_tokenizer.encode_plus(
        agent_prompt, s, max_length=512, truncation=True, return_tensors="np"
    )
    return Intents(np.argmax(twiz_model(**tokens).logits[0]))


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

    print(awnsers)
    best_id = max(range(len(awnsers)), key= lambda x: awnsers[x].score)
    return awnsers[best_id], best_id


class RecipyCategory(Enum):
    General = 0
    Ingredient = 1
    Specific = 2
    



def get_recipy_category(emb):
    return RecipyCategory(recipy_query_svm.predict([emb]))
