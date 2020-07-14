import pandas as pd          # Load library
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\preet\Downloads\Datasets (1)\final data set\areaprice.csv")    # Load Data

# Here - Area--is independent variable
#Here - Price--is dependent variable

from sklearn.linear_model import LinearRegression # after importing LINEAR REGRESSION its performs all features of linear model 
reg = LinearRegression()# now all the functionality of linear regression is transfer in a reg
reg.fit(df[["area"]],df["price"]) #you can also like as reg.fit(df[["area"]],df.price)

reg.predict([[5600]])# asking a question in a form of prediction .. that tell me about the price of given area

reg.score(df[["area"]],df["price"])

# Creating a pickle file for the classifier
import pickle

filename = 'area_price_lr_model.pkl'
pickle.dump(reg, open(filename, 'wb'))









