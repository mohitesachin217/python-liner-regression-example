import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

df = pd.read_csv('Book1.csv')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(us$)')

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
predicted_price = reg.predict(np.array([[3300]]))


with open('model_pickle','wb') as f:
    pickle.dump(reg,f)