import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv('Book1.csv')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(us$)')
plt.scatter(df.area,df.price,color='red',marker='+')
#plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
predicted_price = reg.predict(np.array([[3300]]))


d=pd.read_csv('Book2.csv')

predicted_price = reg.predict(d)

d['prices'] = predicted_price
d.to_csv('prediction.csv',index=False)
print(d)
