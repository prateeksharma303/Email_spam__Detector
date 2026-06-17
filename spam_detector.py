from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk

# load dataset
df = pd.read_csv("emails.csv", encoding="latin-1")

df = df[['spam','text']]
df.columns = ['label','message']

# convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(df["message"])

y = df["label"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = MultinomialNB()
model.fit(X_train, y_train)


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# accuracy
# Predictions

nb_pred = model.predict(X_test)
lr_pred = lr.predict(X_test)

# Probabilities for ROC-AUC

nb_prob = model.predict_proba(X_test)[:,1]
lr_prob = lr.predict_proba(X_test)[:,1]

print("===== Naive Bayes =====")

print("Accuracy:",
      accuracy_score(y_test, nb_pred))

print("Precision:",
      precision_score(y_test, nb_pred))

print("Recall:",
      recall_score(y_test, nb_pred))

print("F1 Score:",
      f1_score(y_test, nb_pred))

print("ROC-AUC:",
      roc_auc_score(y_test, nb_prob))


print("\n===== Logistic Regression =====")

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Precision:",
      precision_score(y_test, lr_pred))

print("Recall:",
      recall_score(y_test, lr_pred))

print("F1 Score:",
      f1_score(y_test, lr_pred))

print("ROC-AUC:",
      roc_auc_score(y_test, lr_prob))
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(scores)

print("Average CV Accuracy:",
      scores.mean())
# Confusion Matrix

cm = confusion_matrix(y_test, nb_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.show()
# Accuracy Comparison Plot

models = [
    "Naive Bayes",
    "Logistic Regression"
]

accuracies = [
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, lr_pred)
]

plt.figure(figsize=(6,4))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.show()
# test message
msg = ["Congratulations you won free lottery"]

msg = cv.transform(msg)

print("Prediction:", model.predict(msg))
window = tk.Tk()
window.title("Spam Email Detector")
def check_spam():
    message = entry.get()
    data = cv.transform([message])
    prediction = model.predict(data)[0]

    if prediction == 1:
        result_label.config(text="Spam Message")
    else:
        result_label.config(text="Not Spam")


# Input box
entry = tk.Entry(window, width=50)
entry.pack(pady=10)

# Button
check_button = tk.Button(window, text="Check Email", command=check_spam)
check_button.pack()

# Result label
result_label = tk.Label(window, text="")
result_label.pack(pady=10)

# Run GUI
window.mainloop()
