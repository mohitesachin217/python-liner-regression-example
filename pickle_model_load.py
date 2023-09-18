import pickle
import numpy as np 

with open('model_pickle','rb') as f:
    mp = pickle.load(f)

input_data = np.array([[5000]])

prediction = mp.predict(input_data)

print("Prediction:", prediction)
