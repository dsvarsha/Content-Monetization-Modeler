import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/youtube_ad_revenue_dataset.csv")
print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())
print(df.describe())

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Histogram
sns.histplot(df["ad_revenue_usd"], kde=True)
plt.title("Revenue Distribution")
plt.savefig("outputs/revenue_hist.png")
plt.close()

print("EDA Completed! Check the outputs folder.")
