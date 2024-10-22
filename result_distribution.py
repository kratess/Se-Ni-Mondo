import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('data/test__results.csv')

# Count the occurrences of each label (you can customize this if necessary)
label_counts = df['Label'].value_counts()

# Plot the distribution using pyplot
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color=['blue', 'green', 'red'])

# Adding labels and title
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(rotation=0)

# Show the plot
plt.show()