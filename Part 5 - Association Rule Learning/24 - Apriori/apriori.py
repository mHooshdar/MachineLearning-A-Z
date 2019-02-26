import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(len(dataset.columns))])
# train
rules = apriori(transactions, min_support=0.003, min_lift=3, min_confidence=0.2, max_length=2)

# visualising
results = list(rules)
