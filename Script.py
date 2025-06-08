# script
import joblib
import numpy as np
# import the model
model = joblib.load('logistic_model.joblib')
# user input
user_input=input("enter the values(separated by comma ',') to predict Heart Disease")
user_input=user_input.split(',')
for i in range(len(user_input)):
  user_input[i]=float(user_input[i])
user_input=np.array(user_input)
print(user_input)
#user_input=np.array()
user_input=user_input.reshape(1,-1)
print(model.predict(user_input))
