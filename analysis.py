import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display the first few rows

print(df.head())

# Explore the structure of the dataset

print(df.info())

# Check for missing values

print(df.isnull().sum())

# Compute basic statistics
print(df.describe())

# Perform groupings on species
print("\nMean Values by Species:")
species_means = df.groupby('species').mean()
print(species_means)

# Insights
print("The mean petal length varies significantly across species.")

import matplotlib.pyplot as plt
import seaborn as sns

# Bar chart
species_means.plot(kind='bar', figsize=(8, 4))
plt.title("Average Feature Values by Species")
plt.xlabel("Species")
plt.ylabel("Average Value")
plt.legend(loc='upper right')
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df[iris.feature_names[2]], kde=True, bins=15)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=iris.feature_names[0], y=iris.feature_names[2], hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.show()



