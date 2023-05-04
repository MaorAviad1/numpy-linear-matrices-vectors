import numpy as np
import pandas as pd

# Load example data
products_revenue = np.array([5000, 12000, 8000, 10000])
X = np.array([1000, 2000, 3000, 4000])
Y = np.array([200, 400, 600, 800])
units_sold = np.array([50, 100, 150, 200])
prices = np.array([100, 90, 80, 70])
sales_matrix = np.array([[50, 100, 150, 200],
                         [40, 80, 120, 160],
                         [30, 60, 90, 120]])

# Define price_matrix
price_matrix = np.array([[100],
                         [90],
                         [80],
                         [70]])

# Calculate results
total_revenue = np.sum(products_revenue)
revenue_percentages = (products_revenue / total_revenue) * 100
A = np.vstack([X, np.ones(len(X))]).T
m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
revenues = np.dot(units_sold, prices)
total_revenue_by_region = np.dot(sales_matrix, price_matrix)

# Create a pandas DataFrame to display the results
product_names = ['Product A', 'Product B', 'Product C', 'Product D']
data = {'Product Name': product_names,
        'Product Revenue': products_revenue,
        'Revenue Percentage': revenue_percentages}
df = pd.DataFrame(data)

# Display the DataFrame
print("\nProduct Revenue Analysis:")
print(df)

# Display additional results
print(f"\nTotal revenue: {total_revenue} USD")
print(f"Linear regression coefficients: m = {m}, c = {c}")
print(f"Dot product of units_sold and prices: {revenues} USD")

# Create a pandas DataFrame for the total revenue by region
region_names = ['Region 1', 'Region 2', 'Region 3']
data = {'Region Name': region_names,
        'Total Revenue': total_revenue_by_region.flatten()}
df_region = pd.DataFrame(data)

# Display the DataFrame
print("\nTotal Revenue by Region:")
print(df_region)
