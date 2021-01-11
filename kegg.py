import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.simplefilter("ignore")

missing_values = ["n/a", "na", "--", "-", "NA"]
df = pd.read_csv("winequality-red.csv", na_values=missing_values)


# sprawdz czy jest jakas cela nie number
# sparwdz czy jest jakas cela pusta

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False


df = df[df.applymap(isnumber)]


# print("Number of unique values in each column:\n")
# for i in df.columns:
#     print(i, len(df[i].unique()))

df['bin_quality'] = pd.cut(df['quality'], bins=[0, 6.5, 10], labels=["bad", "good"])

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

data_length = len(df)
quality_percentage = [100 * i / data_length for i in df["quality"].value_counts()]
bin_quality_percentage = [100 * i / data_length for i in df["bin_quality"].value_counts()]

sns.countplot("quality", data=df, ax=ax[0, 0])
sns.countplot("bin_quality", data=df, ax=ax[0, 1]);

sns.barplot(x=df["quality"].unique(), y=quality_percentage, ax=ax[1, 0])
ax[1, 0].set_xlabel("quality")

sns.barplot(x=df["bin_quality"].unique(), y=bin_quality_percentage, ax=ax[1, 1])
ax[1, 1].set_xlabel("bin_quality")

plt.show()
# for i in range(2):
#     ax[1, i].set_ylabel("The percentage of the total number")
#     ax[1, i].set_yticks(range(0, 101, 10))
#     ax[1, i].set_yticklabels([str(i) + "%" for i in range(0, 101, 10)])
#     for j in range(2):
#         ax[i, j].yaxis.grid()
#         ax[i, j].set_axisbelow(True)

#  kazda kolumne podzielic na przedzialy np od 1 - 2 - 3 - 4 ... i sprawdzic ile celi jestw  tych przedzialach
# podaj min max kazdej kolumny i srednia i czestosc wystepowania poszczegolnych odowiedzi
