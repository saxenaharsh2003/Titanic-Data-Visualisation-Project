import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset from Seaborn's built-in repository
titanic = sns.load_dataset('titanic')

# Data Cleaning: Drop rows with missing 'age'
titanic_clean = titanic.dropna(subset=['age'])

# Set a consistent style
sns.set(style="whitegrid")

# 1. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=titanic_clean, x='sex', hue='survived', palette='muted')
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# 2. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=titanic_clean, x='pclass', hue='survived', palette='pastel')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# 3. Age Distribution: Survivors vs Non-Survivors
plt.figure(figsize=(8,5))
sns.histplot(data=titanic_clean, x='age', hue='survived', kde=True, bins=30, palette='Set2')
plt.title('Age Distribution: Survivors vs Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()
