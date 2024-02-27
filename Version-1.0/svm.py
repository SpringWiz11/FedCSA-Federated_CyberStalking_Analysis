import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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

#SVM model
svm_model = SVC(kernel='linear')  # You can choose different kernels (linear, rbf, poly, etc.)
start_time = time.time()
svm_model.fit(X_train_tfidf, y_train)
end_time = time.time()

#prediction using model
y_pred = svm_model.predict(X_test_tfidf)

#performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
print(f"Time taken to train the SVM model: {end_time - start_time:.4f} seconds")
print("\n")