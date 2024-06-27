import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. Gerekli kütüphanelerin yüklenmesi
import warnings
warnings.filterwarnings('ignore')

# 2. Veri kümesinin yüklenmesi ve bölünmesi
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Model ve hiperparametre aralıklarının tanımlanması
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# 4. Grid Search işleminin gerçekleştirilmesi
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 5. En iyi modelin ve hiperparametrelerin bulunması ve performans değerlendirilmesi
print("En iyi hiperparametreler: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
