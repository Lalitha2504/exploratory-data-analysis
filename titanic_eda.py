# Titanic Data Cleaning & Exploratory Data Analysis (EDA)
# -------------------------------------------------------
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (10, 6)

# -------------------------------
# Step 1: Load the dataset
# -------------------------------
# Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

print("✅ Dataset Loaded Successfully!")
print("Shape of dataset:", titanic.shape)
print("\nFirst 5 rows:")
print(titanic.head())

# -------------------------------
# Step 2: Data Cleaning
# -------------------------------
print("\n🔹 Missing Values:")
print(titanic.isnull().sum())

# Fill missing 'age' with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Fill missing 'embarked' with most common value
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop 'deck' column (too many missing values)
titanic.drop(columns=['deck'], inplace=True)

# Drop rows with missing 'embark_town'
titanic.dropna(subset=['embark_town'], inplace=True)

print("\n✅ After Cleaning Missing Data:")
print(titanic.isnull().sum())

# -------------------------------
# Step 3: Basic Statistics
# -------------------------------
print("\n📊 Summary Statistics:")
print(titanic.describe(include='all'))

# -------------------------------
# Step 4: Exploratory Data Analysis (EDA)
# -------------------------------

# --- (1) Survival Count ---
plt.figure()
sns.countplot(data=titanic, x='survived', palette='Set2')
plt.title('Survival Distribution (0 = Died, 1 = Survived)')
plt.show()

# --- (2) Gender vs Survival ---
plt.figure()
sns.countplot(data=titanic, x='sex', hue='survived', palette='husl')
plt.title('Survival Rate by Gender')
plt.show()

# --- (3) Class vs Survival ---
plt.figure()
sns.countplot(data=titanic, x='class', hue='survived', palette='coolwarm')
plt.title('Survival Rate by Passenger Class')
plt.show()

# --- (4) Age Distribution ---
plt.figure()
sns.histplot(titanic['age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.show()

# --- (5) Age vs Survival ---
plt.figure()
sns.boxplot(data=titanic, x='survived', y='age', palette='muted')
plt.title('Age Distribution by Survival')
plt.show()

# --- (6) Fare Distribution ---
plt.figure()
sns.histplot(titanic['fare'], bins=40, kde=True, color='orange')
plt.title('Fare Distribution')
plt.show()

# --- (7) Correlation Heatmap ---
plt.figure()
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Numeric Variables')
plt.show()

# --- (8) Survival by Embark Town ---
plt.figure()
sns.countplot(data=titanic, x='embark_town', hue='survived', palette='Set3')
plt.title('Survival Rate by Embarkation Town')
plt.show()

# --- (9) Pair Plot for Key Variables ---
sns.pairplot(titanic, vars=['age', 'fare'], hue='survived', palette='husl')
plt.suptitle('Pairwise Relationship of Age, Fare, and Survival', y=1.02)
plt.show()

# -------------------------------
# Step 5: Insights & Observations
# -------------------------------
print("\n🧠 Key Insights:")
print("""
1️⃣ Females had a much higher survival rate than males.
2️⃣ Passengers in 1st class had a higher survival probability.
3️⃣ Younger passengers and children tended to survive more.
4️⃣ Passengers who paid higher fares (wealthier) had better survival chances.
5️⃣ Most people embarked from 'Southampton'.
""")
