import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to check if scraping is allowed
def paths_allowed(url):
    response = requests.get(f"https://www.propertypro.ng/robots.txt")
    if url in response.text:
        return False
    return True

print(paths_allowed("https://www.propertypro.ng/property-for-rent/in/lagos"))

# Scraping data from multiple pages
site_url = "https://www.propertypro.ng/property-for-rent/in/lagos?page="
property_list = []

for page_no in range(0, 6525 + 1):
    page_url = f"{site_url}{page_no}"
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, "html.parser")

    descriptions = [desc.text for desc in soup.select(".listings-property-title2")]
    locations = [loc.text for loc in soup.select("a+ h4")]
    prices = [price.text for price in soup.select(".n50 span:nth-child(2)")]
    beds = [bed.text for bed in soup.select(".fur-areea span:nth-child(1)")]
    baths = [bath.text for bath in soup.select(".fur-areea span:nth-child(2)")]
    toilets = [toilet.text for toilet in soup.select("span~ span+ span")]

    for desc, loc, price, bed, bath, toilet in zip(descriptions, locations, prices, beds, baths, toilets):
        property_list.append([desc, loc, price, bed, bath, toilet])
    
    print(f"page_no: {page_no}")

property_df = pd.DataFrame(property_list, columns=['description', 'location', 'price', 'bed', 'baths', 'toilet'])

# Cleaning the scrapped data
property_df_cleaned = property_df.copy()

# Filtering out apartments with payment per month/day/sqm and price outside the range
property_df_cleaned = property_df_cleaned[~property_df_cleaned['price'].str.contains("/month|/day|/sqm")]
property_df_cleaned['price'] = property_df_cleaned['price'].str.replace("/year", "").str.replace(",", "").astype(float)
property_df_cleaned = property_df_cleaned[(property_df_cleaned['price'] > 20000) & (property_df_cleaned['price'] < 50000000)]

# Cleaning and converting columns
property_df_cleaned['bed'] = property_df_cleaned['bed'].str.replace(" beds", "").astype(int)
property_df_cleaned['baths'] = property_df_cleaned['baths'].str.replace(" baths", "").str.strip().astype(int)
property_df_cleaned['toilet'] = property_df_cleaned['toilet'].str.replace(" Toilets", "").str.strip().astype(int)

# Dropping rows with missing values from bed, baths, and toilet
property_df_cleaned = property_df_cleaned.dropna(subset=['bed', 'baths', 'toilet'])

# Creating "city" column and handling exceptions
property_df_cleaned['city'] = property_df_cleaned['location'].apply(lambda x: x.split()[-2])
property_df_cleaned.loc[property_df_cleaned['city'] == 'Island', 'city'] = 'Victoria Island'
property_df_cleaned.loc[property_df_cleaned['city'] == 'Village', 'city'] = 'Anthony Village'

# Removing non-Lagos locations and non-residential spaces
non_lagos_cities = ["Camp", "Guzape", "Gwarinpa"]
property_df_cleaned = property_df_cleaned[~property_df_cleaned['city'].isin(non_lagos_cities)]
property_df_cleaned = property_df_cleaned[~property_df_cleaned['description'].str.contains("Office|Shop|Hotel|Commercial|Warehouse|Truck|Seminar|Musical|Car")]

# Creating "house_type" and "new" columns
property_df_cleaned['house_type'] = property_df_cleaned['description'].apply(lambda x: 'Flats & Apartments' if re.search('[Ff]lat|[Aa]partment|brf', x) else (
    'Duplex' if re.search('Duplex|Terrace|Detached', x) else (
    'Maisonette' if 'Maison' in x else (
    'Mansion' if 'Mansion' in x else (
    'Self Contain' if 'Self Contain' in x else (
    'Bungalow' if 'Bungalow' in x else (
    'Penthouse' if 'Penthouse' in x else 'Others')))))))

property_df_cleaned['new'] = property_df_cleaned['description'].apply(lambda x: 'New' if re.search('[Nn]ew', x) else 'Old')

# Dropping rows with missing values
property_df_cleaned = property_df_cleaned.dropna()

# Exploring the dataset
print(property_df_cleaned['price'].describe())

# Plotting distribution of prices
plt.figure(figsize=(10, 6))
sns.histplot(property_df_cleaned['price'], bins=10, log_scale=(False, True))
plt.title("Distribution of Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Plotting distribution of prices per city
plt.figure(figsize=(10, 6))
sns.boxplot(x='city', y='price', data=property_df_cleaned)
plt.yscale("log")
plt.xticks(rotation=45)
plt.title("Distribution of Prices per City")
plt.show()

# Plotting distribution of prices per house type
plt.figure(figsize=(10, 6))
sns.boxplot(x='house_type', y='price', data=property_df_cleaned)
plt.yscale("log")
plt.title("Distribution of Prices per House Type")
plt.show()

# Plotting distribution of prices per number of bedrooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='bed', y='price', data=property_df_cleaned)
plt.yscale("log")
plt.title("Distribution of Prices per Number of Bedrooms")
plt.show()

# Plotting distribution of prices for new and old houses
plt.figure(figsize=(10, 6))
sns.boxplot(x='new', y='price', data=property_df_cleaned)
plt.yscale("log")
plt.title("Distribution of Prices For New and Old Houses")
plt.show()

# Checking for significant differences in prices
from scipy.stats import kruskal

print(kruskal(*[group["price"].values for name, group in property_df_cleaned.groupby("city")]))
print(kruskal(*[group["price"].values for name, group in property_df_cleaned.groupby("house_type")]))
print(kruskal(*[group["price"].values for name, group in property_df_cleaned.groupby("bed")]))
print(kruskal(*[group["price"].values for name, group in property_df_cleaned.groupby("new")]))

# Analyzing the data
property_df_cleaned.groupby('bed')['price'].mean().plot(kind='bar')
plt.title("Average Price Per Number of Bedrooms")
plt.ylabel("Average Price")
plt.xlabel("Number of Bedrooms")
plt.show()

property_df_cleaned.groupby('new')['price'].mean().plot(kind='bar')
plt.title("Average Prices for New and Old Buildings")
plt.ylabel("Average Price")
plt.xlabel("Building Type")
plt.show()

# Creating heatmap data
heatmap = property_df_cleaned[['city', 'house_type', 'price']]
heatmap['price'] = np.log(heatmap['price'])

# Pivoting data for heatmap
heatmap_pivot = heatmap.pivot_table(values='price', index='city', columns='house_type', aggfunc='mean')

# Plotting heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Average Log Prices of House Types in Different Cities")
plt.show()

# Exporting data
property_df.to_csv("scrapped_propertypro_lagos.csv", index=False)
property_df_cleaned.to_csv("propertypro_lagos.csv", index=False)
heatmap.to_csv("heatmap.csv", index=False)
