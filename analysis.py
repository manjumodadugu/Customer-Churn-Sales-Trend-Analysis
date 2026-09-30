import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
df = pd.read_excel(URL, sheet_name="Year 2009-2010")

df.columns = [c.strip().replace(" ", "_") for c in df.columns]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df = df.dropna(subset=["Customer_ID", "InvoiceDate", "Quantity", "Price"])
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].drop_duplicates()

# Feature engineering
df["Revenue"] = df["Quantity"] * df["Price"]
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
df["Quarter"] = df["InvoiceDate"].dt.to_period("Q").astype(str)
df["Weekday"] = df["InvoiceDate"].dt.day_name()

# Customer metrics
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
customer = df.groupby("Customer_ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency=("InvoiceDate", "nunique"),
    Monetary=("Revenue", "sum"),
    Orders=("Invoice", "nunique")
).reset_index()

customer["Segment"] = "Regular"
customer.loc[
    (customer["Monetary"] >= customer["Monetary"].quantile(.75)) &
    (customer["Recency"] <= customer["Recency"].quantile(.25)),
    "Segment"
] = "High Value"
customer.loc[
    (customer["Monetary"] < customer["Monetary"].quantile(.25)) &
    (customer["Recency"] > customer["Recency"].quantile(.75)),
    "Segment"
] = "At Risk"
customer.loc[
    (customer["Frequency"] >= customer["Frequency"].quantile(.75)) &
    (customer["Monetary"] >= customer["Monetary"].quantile(.50)),
    "Segment"
] = "Loyal"

# Pivot tables
monthly_sales = pd.pivot_table(df, index="Month", values="Revenue", aggfunc="sum")
segment_sales = customer.groupby("Segment").agg(
    Customers=("Customer_ID", "count"),
    Revenue=("Monetary", "sum"),
    Avg_Revenue=("Monetary", "mean")
).sort_values("Revenue", ascending=False)

print("Monthly revenue:")
print(monthly_sales)
print("\nCustomer segments:")
print(segment_sales)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10,5))
monthly_sales["Revenue"].plot(marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(OUT / "monthly_revenue_trend.png", dpi=150)
plt.close()

plt.figure(figsize=(9,5))
sns.barplot(data=segment_sales.reset_index(), x="Segment", y="Revenue",
            hue="Segment", legend=False)
plt.title("Revenue by Customer Segment")
plt.tight_layout()
plt.savefig(OUT / "segment_revenue.png", dpi=150)
plt.close()

top_products = df.groupby("Description")["Revenue"].sum().nlargest(10).sort_values()
plt.figure(figsize=(9,6))
top_products.plot(kind="barh")
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(OUT / "top_products.png", dpi=150)
plt.close()

country_sales = df.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()
plt.figure(figsize=(9,6))
country_sales.plot(kind="barh")
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(OUT / "top_countries.png", dpi=150)
plt.close()
