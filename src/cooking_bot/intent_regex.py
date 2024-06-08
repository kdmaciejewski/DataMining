from cooking_bot.intents import Intents
import re
from nltk.tokenize import wordpunct_tokenize
from typing import Optional


match_patterns = {
    Intents.NextStepIntent: ["next", "go on", "move on"],
    Intents.PreviousStepIntent: ["last", "back", "before"],
    Intents.RepeatIntent: ["repeat", "explain again"],
    Intents.StopIntent: ["stop", "halt", "end", "cancel"],
    Intents.NoIntent: ["no", "nope", "nah"],
    Intents.YesIntent: [
        "yes",
        "yep",
        "sure",
        "ok",
        "okay",
        "allright",
    ],
}


def intent_regex(text: str) -> Optional[Intents]:

    words = [i.lower() for i in wordpunct_tokenize(text) if i.isalpha()]

    for intent, patterns in match_patterns.items():

        for pattern in patterns:
            split = pattern.split()

            for token in words:
                if token == split[0]:
                    split.pop(0)

                if len(split) == 0:
                    return intent

    return None


if __name__ == "__main__":

    texts = [
        "yes please",
        "Nope",
        "what is the next step",
        "can we stop?",
        "go to the last one",
        "okay then",
        "Nah",
        "next step please",
    ]
    for text in texts:
        result = intent_regex(text)
        print(f"'{text}' detected as: {result}")
