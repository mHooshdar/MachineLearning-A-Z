import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

dataset = pd.read_csv('./Ads_CTR_Optimisation.csv')

# random selection
# N = 10000
# d = 10
# ads_selected = []
# total_reward = 0
# for n in range(N):
#     ad = random.randrange(d)
#     ads_selected .append(ad)
#     reward = dataset.values[n, ad]
#     total_reward = total_reward + reward


# ucb
N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if number_of_selections[i] > 0:
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt((3 * math.log(n + 1)) / (2 * number_of_selections[i]))
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Histogram random')
plt.xlabel('ads')
plt.ylabel('Number of time each selected')
plt.show()
