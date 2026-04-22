import pandas as pd
import numpy as np
import re
import warnings
warnings.simplefilter("ignore")

# Load dataset
data = pd.read_csv("Language Detection.csv")

# Split features and labels
X = data["Text"]
y = data["Language"]

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Text preprocessing
data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?:;~`0-9]', ' ', text)
    text = text.lower()
    data_list.append(text)

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list)

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model training
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Prediction function
def predict(text):
    text = re.sub(r'[!@#$(),\n"%^*?:;~`0-9]', ' ', text)
    text = text.lower()
    x = cv.transform([text])
    lang = model.predict(x)
    print("Language:", le.inverse_transform(lang)[0])

predict("bonjour")
