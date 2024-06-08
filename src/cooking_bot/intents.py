from enum import Enum
from pydantic import BaseModel

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
    StopIntent = 35
    
    
    
class RecipyCategory(Enum):
    LookBased = 0
    Ingredient = 1
    Specific = 2
    General = 3
    
    
class TimeCategory(Enum):
    No = 0
    Specific = 1
    Urgent = 2
    
class Difficulty(Enum):
    No = 0
    Easy = 1
    

class RecipyIntent(BaseModel):
    category : RecipyCategory
    time : TimeCategory
    diff : Difficulty
    

    