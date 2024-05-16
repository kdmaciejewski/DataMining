from tqdm import tqdm
from cooking_bot.encoders import get_sentence_embedding
import json
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


with open("datasets/recipe_decision.json") as f:
    example_queries = json.load(f)

inputs, labels = zip(*example_queries)

embeddings = np.array([get_sentence_embedding(text) for text in tqdm(inputs)])




# Assume 'embeddings' and 'labels' are already defined and imported correctly
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
print(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, encoded_labels, test_size=0.2, random_state=42
)
# Initialize additional classifiers
svm_classifier = SVC()

# Train additional classifiers
svm_classifier.fit(X_train, y_train)

# Predict using Gradient Boosting

# Predict using SVM
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"Support Vector Machine Accuracy: {svm_accuracy}")


with open("models/recipe_query_svm", 'wb') as f:
    pickle.dump(svm_classifier, f)