import pandas as pd           # handling tables
import numpy as np            # math operations
import matplotlib.pyplot as plt # drawing graphs
import seaborn as sns         # better graphs
from sklearn.model_selection import train_test_split # splits data
from sklearn.linear_model import LogisticRegression  # algorithm
from sklearn.metrics import accuracy_score, confusion_matrix # graders


df = pd.read_csv('diabetes.csv')

missing_data_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# replaces 0 with nan
df[missing_data_cols] = df[missing_data_cols].replace(0, np.nan)

df.fillna(df.mean(), inplace=True)

print("Data loaded and cleaned successfully.")


X = df.drop('Outcome', axis=1)
y = df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training the model...")
model = LogisticRegression(max_iter=1000) # extends time
model.fit(X_train, y_train)             # EPOCH = round


print("Evaluating...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\n[True Neg  False Pos]")
print("[False Neg True Pos]")