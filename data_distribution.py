import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the data
data = pd.read_csv('epoch_20960.csv')

# Extract the 'Weight' column
weights = data['Weight']

# Set up the plot style
sns.set(style="whitegrid")

# Step 2: Plot Histogram with KDE (to visualize distribution)
plt.figure(figsize=(10, 6))
sns.histplot(weights, kde=True, bins=20, color='skyblue')
plt.title('Histogram with KDE of Weight')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

# Step 3: Q-Q plot (to check normality visually)
plt.figure(figsize=(6, 6))
stats.probplot(weights, dist="norm", plot=plt)
plt.title('Q-Q Plot of Weight')
plt.show()