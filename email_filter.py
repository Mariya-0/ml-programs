from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample email data
emails = [
    "Free money now!!!",
    "Hello, how are you doing today?",
    "Get a loan today with low interest!",
    "Meeting at 2pm today in the conference room.",
    "You won a lottery! Claim your prize now!",
    "Let's catch up for coffee later.",
    "Buy cheap medicines online now!",
    "Please find the attached report for the project."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham

# Convert text data to numeric vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test new message
new_email = ["Claim your free prize now! You won a million dollars!"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)

print("Spam" if prediction[0] == 1 else "Ham")
