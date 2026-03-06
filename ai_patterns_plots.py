import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



fsize = 16

# Data extracted from the confusion matrix
data = [
    [6, 1, 0, 4, 2, 1, 4, 2],
    [0, 50, 6, 0, 2, 0, 1, 4],
    [0, 6, 38, 0, 3, 0, 0, 5],
    [2, 1, 0, 10, 1, 1, 3, 2],
    [0, 4, 8, 0, 25, 3, 1, 9],
    [0, 0, 0, 1, 4, 18, 2, 3],
    [2, 0, 1, 5, 1, 0, 11, 11],
    [2, 10, 8, 2, 9, 4, 4, 46]
]

# Labels for the classes
labels = [f"C{i}" for i in range(1, 9)]

# Create a DataFrame for better plotting
df_cm = pd.DataFrame(data, index=labels, columns=labels)

# Plotting the heatmap
plt.figure(figsize=(10, 7))
#set the font size of the heatmap values to 16
sns.set(font_scale=1.5)
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True)


# Adding labels and title
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix Heatmap')
#increase the font size of the labels
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.title('Confusion Matrix Heatmap', fontsize=fsize)
plt.xlabel('Predicted Class', fontsize=fsize)
plt.ylabel('True Class', fontsize=fsize)

# Display or save the plot
#plt.show()
#save as pdf
plt.savefig('confusion_matrix.pdf')



import matplotlib.pyplot as plt
import pandas as pd

# Data from the provided table
data = {
    'Class': [
        'C3-Multimodal', 'C2-Classical', 'C1-Agent', 
        'C5-Prepro-TN', 'C6-RAG', 'C4-Abstraction', 'C7-Tools'
    ],
    'Forecast': [246, 259, 60, 166, 71, 34, 61],
    'P5': [246, 245, 55, 55, 50, 0, 0],
    'P95': [330, 321, 114, 143, 92, 40, 0]
}

df = pd.DataFrame(data)

# Calculate error values (distance from Forecast to P5 and P95)
# If P5/P95 are the bounds, error is |Forecast - Bound|
lower_error = df['Forecast'] - df['P5']
upper_error = df['P95'] - df['Forecast']
asymmetric_error = [ [x if x > 0 else 0 for x in lower_error], [x if x > 0 else 0 for x in upper_error]]

# Create the plot
plt.figure(figsize=(10, 6))

# Use errorbar to create the confidence interval plot
plt.errorbar(
    df['Forecast'], df['Class'], 
    xerr=asymmetric_error, 
    fmt='o',          # Circular marker for the forecast point
    color='teal',     # Color of markers and bars
    ecolor='gray',    # Color of error bars
    elinewidth=2,     # Thickness of error bars
    capsize=5,        # Size of the horizontal caps at the ends
    label='Forecast (with P5-P95 Interval)'
)

# Formatting
plt.title('Confidence Intervals (P5 to P95) for Class Forecasts', fontsize=fsize)
plt.xlabel('Occurrence', fontsize=fsize)
plt.ylabel('Pattern Class', fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()  # Invert so the top class in the table is at the top of the plot
plt.tight_layout()

# Show the plot
#plt.show()
plt.savefig('confidence_intervals.pdf')