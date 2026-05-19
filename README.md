# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and preprocess the spam dataset by converting labels and extracting message features using TF-IDF vectorization.
2. Split the dataset into training and testing data for model evaluation.
3. Train the SVM classifier using the training dataset and predict spam or ham messages on test data.
4. Generate a confusion matrix and visualize the classification results using a heatmap.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Elakya L
RegisterNumber: 212225230066
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1','v2']]
data.columns = ['label','message']
data['label'] = data['label'].map({'ham':0, 'spam':1})
X = data['message']
y = data['label']
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show()
```

## Output:
<img width="576" height="500" alt="WhatsApp Image 2026-05-19 at 8 56 44 AM" src="https://github.com/user-attachments/assets/f5be8f4a-5691-417e-9a44-ce1297764325" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
