import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("../data/superstore.csv", encoding='latin1')

# -------------------------------
# CLEAN COLUMN NAMES
# -------------------------------
df.columns = df.columns.str.encode('ascii', 'ignore').str.decode('ascii')
df.columns = df.columns.str.replace('.', ' ')
df.columns = df.columns.str.strip()

print("Columns:\n", df.columns)

# -------------------------------
# FIX DATE COLUMN (IMPORTANT 🔥)
# -------------------------------
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Remove invalid dates
df = df.dropna(subset=['Order Date'])

# -------------------------------
# CLEAN DATA
# -------------------------------
df = df.dropna()

# -------------------------------
# BASIC ANALYSIS
# -------------------------------

# Total Sales
total_sales = df['Sales'].sum()
print("Total Sales:", total_sales)

# Sales by Region
region_sales = df.groupby('Region')['Sales'].sum()

# Sales by Category
category_sales = df.groupby('Category')['Sales'].sum()

# Top Products
top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(5)

# -------------------------------
# ADVANCED ANALYSIS 🔥
# -------------------------------

# Monthly Sales Trend
df['Month'] = df['Order Date'].dt.month
monthly_sales = df.groupby('Month')['Sales'].sum()

# Top Customers
top_customers = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(5)

# Profit by Region
profit_region = df.groupby('Region')['Profit'].sum()

# -------------------------------
# SAVE RESULTS
# -------------------------------
with open("../outputs/results.txt", "w") as f:
    f.write(f"Total Sales: {total_sales}\n\n")
    f.write("Sales by Region:\n")
    f.write(str(region_sales))
    f.write("\n\nTop Products:\n")
    f.write(str(top_products))

# -------------------------------
# VISUALIZATIONS
# -------------------------------

# Region Sales
region_sales.plot(kind='bar', title="Sales by Region")
plt.savefig("../outputs/charts/region_sales.png")
plt.close()

# Category Sales
category_sales.plot(kind='bar', title="Sales by Category")
plt.savefig("../outputs/charts/category_sales.png")
plt.close()

# Top Products
top_products.plot(kind='bar', title="Top Products")
plt.savefig("../outputs/charts/top_products.png")
plt.close()

# Monthly Sales
monthly_sales.plot(kind='line', marker='o', title="Monthly Sales Trend")
plt.savefig("../outputs/charts/monthly_sales.png")
plt.close()

# Top Customers
top_customers.plot(kind='bar', title="Top Customers")
plt.savefig("../outputs/charts/top_customers.png")
plt.close()

# Profit Region
profit_region.plot(kind='bar', title="Profit by Region")
plt.savefig("../outputs/charts/profit_region.png")
plt.close()

print("\n✅ Project Completed Successfully!")