from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
digits = load_digits()
plt.gray() 
for i in range(5):
    plt.matshow(digits.images[i]) 
dir(digits)
digits.data[0]
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)
model.fit(X_train, y_train)
model.score(X_test, y_test)
predicted = model.predict(digits.data[0:5])
print(predicted)

