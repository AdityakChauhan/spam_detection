import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import joblib

# Load your dataset
df = pd.read_csv("D:/FOT/sem4/fda/project/spam_detection/dataset/processed/dataset.csv", encoding='latin-1')
df.columns = ['label', 'text']

# Convert text column to string type to prevent errors
df['text'] = df['text'].astype(str)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Create a pipeline with TF-IDF and LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(C=1, max_iter=1000, random_state=42))
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Save the model
joblib.dump(pipeline, 'spam_classifier_model.pkl')
print("Model saved as 'spam_classifier_model.pkl'")

# Example of how to use the saved model
print("\nExample prediction:")
test_messages = [
    "Congratulations! You've won a free iPhone! Click here to claim your prize now!",
    "Hi, can you please send me the meeting notes from yesterday? Thanks!"
]
predictions = pipeline.predict(test_messages)
for msg, pred in zip(test_messages, predictions):
    print(f"Message: {msg[:50]}{'...' if len(msg) > 50 else ''}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}\n")