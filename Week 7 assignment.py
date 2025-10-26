# Analyzing Data with Pandas and Visualizing Results with Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 70)
print("DATA ANALYSIS WITH PANDAS AND MATPLOTLIB")
print("=" * 70)

# Task 1: Load and Explore the Dataset
print("\n" + "="*50)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("="*50)

try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Display first few rows
    print("\n📋 First 5 rows of the dataset:")
    print(df.head())
    
    # Display basic information
    print("\n📊 Dataset Info:")
    print(df.info())
    
    # Check for missing values
    print("\n🔍 Missing Values Check:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # Since Iris dataset is clean, let's demonstrate cleaning with a hypothetical scenario
    print("\n🧹 Data Cleaning:")
    if df.isnull().sum().sum() == 0:
        print("No missing values found. Dataset is already clean!")
    else:
        # This would execute if there were missing values
        df_cleaned = df.dropna()  # or df.fillna(method='ffill')
        print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n" + "="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*50)

# Basic statistics
print("📈 Basic Statistics for Numerical Columns:")
print(df.describe())

# Species-specific statistics
print("\n🌿 Species-specific Statistics:")
species_stats = df.groupby('species').describe()
print(species_stats)

# Group by species and compute mean for each numerical column
print("\n📊 Mean Values by Species:")
species_means = df.groupby('species').mean()
print(species_means)

# Additional analysis - correlation
print("\n🔗 Correlation Matrix:")
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print(correlation_matrix)

# Interesting findings
print("\n💡 INTERESTING FINDINGS:")
print("1. Setosa species has significantly smaller petal dimensions")
print("2. Virginica has the largest petal length and width on average")
print("3. Sepal dimensions show more overlap between species")
print("4. Strong positive correlation between petal length and petal width")

# Task 3: Data Visualization
print("\n" + "="*50)
print("TASK 3: DATA VISUALIZATION")
print("="*50)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Line Chart - Trends in measurements (using index as pseudo-time)
plt.subplot(2, 3, 1)
plt.plot(df.index[:50], df['sepal length (cm)'][:50], marker='o', label='Sepal Length', linewidth=2)
plt.plot(df.index[:50], df['petal length (cm)'][:50], marker='s', label='Petal Length', linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel('Length (cm)')
plt.title('Line Chart: Sepal vs Petal Length Trends\n(First 50 samples)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Bar Chart - Average measurements by species
plt.subplot(2, 3, 2)
species_avg = df.groupby('species').mean()
x = np.arange(len(species_avg.index))
width = 0.2

plt.bar(x - width, species_avg['sepal length (cm)'], width, label='Sepal Length', alpha=0.8)
plt.bar(x, species_avg['petal length (cm)'], width, label='Petal Length', alpha=0.8)
plt.bar(x + width, species_avg['sepal width (cm)'], width, label='Sepal Width', alpha=0.8)

plt.xlabel('Species')
plt.ylabel('Average Measurement (cm)')
plt.title('Bar Chart: Average Measurements by Species', fontsize=12, fontweight='bold')
plt.xticks(x, species_avg.index, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Histogram - Distribution of sepal length
plt.subplot(2, 3, 3)
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram: Distribution of Sepal Length', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Scatter Plot - Sepal length vs Petal length colored by species
plt.subplot(2, 3, 4)
colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}

for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
                species_data['petal length (cm)'], 
                c=colors[species], 
                label=species, 
                alpha=0.7, 
                s=60)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Scatter Plot: Sepal vs Petal Length by Species', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Additional Plot 1: Box plot by species
plt.subplot(2, 3, 5)
df.boxplot(column=['sepal length (cm)', 'petal length (cm)'], by='species', grid=True)
plt.suptitle('')  # Remove automatic title
plt.title('Box Plot: Length Distribution by Species', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)

# 6. Additional Plot 2: Correlation heatmap
plt.subplot(2, 3, 6)
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Heatmap: Feature Correlations', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Additional detailed visualizations
print("\n📊 ADDITIONAL DETAILED VISUALIZATIONS")

# Pairplot using seaborn
plt.figure(figsize=(12, 8))
sns.pairplot(df, hue='species', diag_kind='hist', palette='husl')
plt.suptitle('Pairplot: Feature Relationships by Species', y=1.02, fontweight='bold')
plt.show()

# Violin plots for detailed distribution analysis
plt.figure(figsize=(15, 10))

features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.violinplot(x='species', y=feature, data=df, palette='husl')
    plt.title(f'Violin Plot: {feature} by Species', fontweight='bold')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Summary of findings
print("\n" + "="*70)
print("SUMMARY OF FINDINGS AND OBSERVATIONS")
print("="*70)

print("\n🔍 KEY INSIGHTS FROM THE IRIS DATASET ANALYSIS:")
print("1. SPECIES DIFFERENTIATION:")
print("   - Setosa is clearly distinguishable with small petals")
print("   - Versicolor and Virginica show some overlap in measurements")
print("   - Petal measurements are better for species classification")

print("\n2. MEASUREMENT PATTERNS:")
print("   - Strong positive correlation between petal length and width (0.96)")
print("   - Moderate correlation between sepal and petal length (0.87)")
print("   - Sepal width shows weakest correlations with other features")

print("\n3. DISTRIBUTION CHARACTERISTICS:")
print("   - Sepal length shows approximately normal distribution")
print("   - Petal measurements show bimodal distribution due to species differences")
print("   - Setosa has the most compact distribution across all features")

print("\n4. VISUALIZATION EFFECTIVENESS:")
print("   - Scatter plots effectively show species clusters")
print("   - Box plots highlight statistical differences between species")
print("   - Violin plots provide detailed distribution insights")

print("\n" + "="*70)
print("ASSIGNMENT COMPLETED SUCCESSFULLY! 🎉")
print("="*70)