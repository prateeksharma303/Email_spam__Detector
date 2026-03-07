import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
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

# accuracy
print("Accuracy:", model.score(X_test,y_test))

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