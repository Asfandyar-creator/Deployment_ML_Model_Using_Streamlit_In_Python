import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


input_data = (3,107,62,13,48,22.9,0.678,23)

# Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the input data as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The female is Non-diabetic.")
else:
    print("The female is diabetic.")

