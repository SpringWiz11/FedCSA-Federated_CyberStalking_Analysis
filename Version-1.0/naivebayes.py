import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#import dataset
dataset_path = '/home/kishan/Documents/projects/machinelearning_cyberstalking_research/dataset.csv'
df = pd.read_csv(dataset_path)

#performing data pre processing by converting to lowercase
df['text'] = df['text'].str.replace('[^a-zA-Z\s]', '').str.lower()

#split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

#feature extraction using TF-IDF method
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Naive Bayes model
nb_model = MultinomialNB()
start_time = time.time()
nb_model.fit(X_train_tfidf, y_train)
end_time = time.time()

#prediction using model
y_pred = nb_model.predict(X_test_tfidf)
probability_scores = nb_model.predict_proba(X_test_tfidf)  # Added line for probability scores

#performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

#probability scores for the first few predictions
print("\nProbability Scores for the first few predictions:")
print("Probability score here is represented in values for [non-cyberstalking, cyberstalking]\n")
for i in range(6, min(9, len(X_test))):
    print(f"Text: {X_test.iloc[i]}")
    print(f"True Label: {y_test.iloc[i]}, Predicted Label: {y_pred[i]}")
    print(f"Probability Scores: {probability_scores[i]}")
    print()

print(f"Time taken to train the Naive Bayes Model: {end_time - start_time:.4f} seconds")