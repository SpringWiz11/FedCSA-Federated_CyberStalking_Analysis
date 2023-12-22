import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#import dataset
dataset_path = '/Users/sameera/Downloads/dataset.csv'
df = pd.read_csv(dataset_path)

#performing data pre processing by converting to lowercase
df['text'] = df['text'].str.replace('[^a-zA-Z\s]', '').str.lower()

#split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

#feature extraction using TF-IDF method
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Random Forest model
rf_model = RandomForestClassifier()
start_time = time.time()
rf_model.fit(X_train_tfidf, y_train)
end_time = time.time()

#prediction using model
y_pred = rf_model.predict(X_test_tfidf)

#performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Time taken to train the Random Forest model: {end_time - start_time:.4f} seconds")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
