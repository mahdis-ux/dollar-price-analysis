import pandas as pd
df = pd.read_csv("dollar.csv.txt")
print("اطلاعات:")
print(df.head())
average = df["price"].mean()
print("میانگین قیمت دلار: ", average)
max_price=df["price"].max()
print("بیشترین قیمت: ", max_price)
min_price=df["price"].max()
print("کمترین قیمت: ", min_price)

import matplotlib.pyplot as plt
plt.plot(df["date"], df["price"])
plt.xlabel("date")
plt.ylabel("price")
plt.title("Dollar price trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
