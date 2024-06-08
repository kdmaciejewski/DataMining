from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
from cooking_bot import REPO_PATH
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from loguru import logger
from cooking_bot.intent_regex import intent_regex
from cooking_bot.intents import (
    Intents,
    TimeCategory,
    Difficulty,
    RecipyCategory,
    RecipyIntent,
)
from cooking_bot.encoders import get_sentence_embedding


class IntentDetector:

    regex_cutoff = 20

    def __init__(self) -> None:

        self.twiz_model = ORTModelForSequenceClassification.from_pretrained(
            REPO_PATH / "models/twiz-intent"
        )
        self.twiz_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        with open(str(REPO_PATH / "models/recipe_query_mlp"), "rb") as f:
            self.recipy_query_mlp: MLPClassifier = pickle.load(f)

        with open(str(REPO_PATH / "models/diff_mlp"), "rb") as f:
            self.difficulty_mlp: MLPClassifier = pickle.load(f)

        with open(str(REPO_PATH / "models/time_mlp"), "rb") as f:
            self.time_mlp: MLPClassifier = pickle.load(f)

    def get_intent(
        self,
        s: str,
        agent_prompt: str = "I can help you finding delicious recipes. What kind of recipe would you like to search for?",
    ) -> Intents:

        if len(s) < self.regex_cutoff:

            res = intent_regex(s)

            logger.debug(f"Matched with regex to '{s}' - {res}")

            if res is not None:
                return res

        # default back to twiz
        tokens = self.twiz_tokenizer.encode_plus(
            agent_prompt, s, max_length=512, truncation=True, return_tensors="np"
        )
        
        tokens = {key : val.astype(np.int64) for key,val in tokens.items()}

        res = Intents(np.argmax(self.twiz_model(**tokens).logits[0]))
        logger.debug(f"Matched with twiz to '{s}' - {res}")

        return res

    def get_recipy_intent(self, s: str) -> RecipyIntent:

        emb = get_sentence_embedding(s)

        def get_class(mlp, cls):

            pred = mlp.predict([emb])
            return cls(pred[0])

        cat_pred = get_class(self.recipy_query_mlp, RecipyCategory)
        time_pred = get_class(self.time_mlp, TimeCategory)
        diff_pred = get_class(self.difficulty_mlp, Difficulty)

        logger.debug(
            f"Recipy Intent Prediction: {cat_pred} - {time_pred} - {diff_pred}"
        )

        return RecipyIntent(category=cat_pred, time=time_pred, diff=diff_pred)


if __name__ == "__main__":

    intent_detector = IntentDetector()

    while True:

        request = input("\nIntent: ")

        intent = intent_detector.get_intent(request)

        if intent in [
            Intents.SelectIntent,
            Intents.SuggestionsIntent,
            Intents.IdentifyProcessIntent,
        ]:
            intent_detector.get_recipy_intent(request)
