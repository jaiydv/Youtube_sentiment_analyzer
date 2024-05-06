import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv")
label= data.values.tolist()

first_column = data.iloc[:, 0].tolist()  # Extract values of the first column
second_column = data.iloc[:, 1].tolist() 
print(first_column)
print(second_column)
# print(values) 