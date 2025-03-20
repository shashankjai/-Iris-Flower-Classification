# -Iris-Flower-Classification
# Iris Flower Classification

## Project Overview
This project classifies iris flowers into three species (*Setosa, Versicolor, Virginica*) using machine learning. The dataset contains features like sepal length, sepal width, petal length, and petal width. The goal is to train a model that can accurately classify an iris flower based on these features.

## Dataset
- **Source**: `sklearn.datasets`
- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target Labels**:
  - Setosa (0)
  - Versicolor (1)
  - Virginica (2)

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Steps
1. **Load Dataset**: Import and load the Iris dataset.
2. **Exploratory Data Analysis (EDA)**: Analyze data distributions with graphs.
3. **Preprocessing**: Split data into training and testing sets.
4. **Train Model**: Use a classifier (e.g., Decision Tree, Logistic Regression).
5. **Evaluate Model**: Measure accuracy and visualize the confusion matrix.
6. **Make Predictions**: Test the model on new data.

## Data Visualization
- **Pairplot**: Shows relationships between features.
- **Feature Distributions**: Histograms for each feature.
- **Confusion Matrix**: Evaluates model performance.

## Example Graph: Pairplot
```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load Dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# Replace target values with species names
df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Plot Pairplot
sns.pairplot(df, hue='species', diag_kind='kde')
plt.show()
```

## Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np

# Split Data
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.2f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Results
- The accuracy of the model is typically **above 90%**.
- The confusion matrix shows the classification performance.

## Future Improvements
- Try different classifiers (SVM, Random Forest, etc.).
- Implement hyperparameter tuning for better accuracy.
- Deploy the model using Flask or Streamlit.

## Conclusion
This project demonstrates basic machine learning concepts, from data exploration to model evaluation. It serves as an excellent starting point for beginners in ML.

---

