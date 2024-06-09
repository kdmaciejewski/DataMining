from tqdm import tqdm
from cooking_bot.encoders import get_sentence_embedding
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neural_network import MLPClassifier
from cooking_bot import REPO_PATH
import os
os.chdir(REPO_PATH)

df = pd.read_csv("datasets/new_clean.csv", index_col=0)

embeddings = np.array([get_sentence_embedding(text) for text in tqdm(df.text)])
labels = df.queries

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)


mlp_class = MLPClassifier(random_state=0, max_iter=300)
mlp_class.fit(X_train, y_train)

mlp_predictions = mlp_class.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

with open("models/recipe_query_mlp", 'wb') as f:
    pickle.dump(mlp_class, f)

print(f"Query Acc: {mlp_accuracy}")



spec = pd.read_csv("datasets/spec.csv", index_col=0)
df = pd.concat((df, spec))
embeddings = np.array([get_sentence_embedding(text) for text in tqdm(df.text)])


X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df.time, test_size=0.2, random_state=42
)




mlp_class = MLPClassifier(random_state=0, max_iter=300)
mlp_class.fit(X_train, y_train)
mlp_predictions = mlp_class.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)


with open("models/time_mlp", 'wb') as f:
    pickle.dump(mlp_class, f)
print(f"Time Acc: {mlp_accuracy}")


X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df.difficulty, test_size=0.2, random_state=42
)


mlp_class = MLPClassifier(random_state=0, max_iter=300)
mlp_class.fit(X_train, y_train)
mlp_predictions = mlp_class.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

with open("models/diff_mlp", 'wb') as f:
    pickle.dump(mlp_class, f)

print(f"difficulty Acc: {mlp_accuracy}")

