import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


sns.set_style("whitegrid")
sns.set_context("talk")


np.random.seed(42)
customers = 500
df = pd.DataFrame({
    "Visits": np.random.poisson(10, customers),
    "Purchases": np.random.poisson(2, customers),
    "SessionTime": np.random.normal(5, 1, customers),
    "EmailOpens": np.random.poisson(15, customers),
    "RepeatPurchases": np.random.binomial(2, 0.2, customers),
    "SurveyScore": np.random.uniform(0, 10, customers)
})


df["Purchases"] += (0.2 * df["Visits"]).astype(int)
df["RepeatPurchases"] += (0.5 * df["Purchases"]).astype(int)
df["SurveyScore"] = np.clip(df["SurveyScore"] + 0.5 * df["RepeatPurchases"], 0, 10)

# Calculate correlation matrix
corr = df.corr()

# Plot heatmap
plt.figure(figsize=(8, 8))  
ax = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",  
    linewidths=0.4,
    square=True,
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title("Customer Engagement Correlation Matrix", fontsize=18, fontweight='bold', pad=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig('chart.png', dpi=64, bbox_inches='tight')  
plt.show()
