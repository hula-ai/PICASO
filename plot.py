import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_grouped = pd.read_excel('./picaso.xlsx')

fig, ax = plt.subplots()
x = range(5,105,5)
ax.plot(x, df_grouped['mean'])
ax.fill_between(
    x, df_grouped['min'], df_grouped['max'], color='b', alpha=.15)

ax.set_title('Avg Taxi Fare by Date')
#fig.autofmt_xdate(rotation=45)

df_grouped = pd.read_excel('./deep.xlsx')
ax.plot(x, df_grouped['mean'])
ax.fill_between(
    x, df_grouped['min'], df_grouped['max'], color='r', alpha=.15)


df_grouped = pd.read_excel('./set.xlsx')
ax.plot(x, df_grouped['mean'])
ax.fill_between(
    x, df_grouped['min'], df_grouped['max'], color='g', alpha=.15)

plt.show()