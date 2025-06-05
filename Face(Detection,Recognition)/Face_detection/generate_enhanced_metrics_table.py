import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import os

# Create the data from the updated metrics
data = {
    'Model': ['Haar Cascade', 'MMOD', 'YOLOv5-Face', 'SSD Face', 'MobileNet SSD'],
    'Accuracy (%)': [99.42, 100.00, 100.00, 100.00, 100.00],
    'Precision (%)': [99.42, 100.00, 100.00, 100.00, 100.00],
    'False Positive Rate': [0.0058, 0.0000, 0.0000, 0.0000, 0.0000],
    'Avg Time (s)': [0.0122, 0.5217, 0.0262, 0.0089, 0.0090],
    'Speed (faces/sec)': [82.10, 1.92, 38.11, 112.14, 110.59]
}

df = pd.DataFrame(data)

# Create a function to highlight cells based on values (better values are more saturated)
def apply_color_scale(val, col_name):
    if col_name == 'Accuracy (%)' or col_name == 'Precision (%)' or col_name == 'Speed (faces/sec)':
        # For these metrics, higher is better
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        # Scale to 0-1 range
        normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        # Use a color scale (green with increasing intensity)
        color = plt.cm.Greens(0.3 + 0.7 * normalized)
        return color
    
    elif col_name == 'False Positive Rate' or col_name == 'Avg Time (s)':
        # For these metrics, lower is better
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        # Scale to 0-1 range
        normalized = 1.0 - ((val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
        # Use a color scale (blue with increasing intensity)
        color = plt.cm.Blues(0.3 + 0.7 * normalized)
        return color
    
    # Default return (for model names etc.)
    return plt.cm.Greys(0.1)

# Create the figure and table
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    colWidths=[0.15, 0.14, 0.14, 0.18, 0.14, 0.18],
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1]
)

# Customize table appearance
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.5)  # Make the table larger

# Style header cells
for i, key in enumerate(df.columns):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold', color='white')
    cell.set_facecolor('#2980b9')  # Dark blue for headers

# Apply colors to data cells
for i in range(len(df)):
    for j, col in enumerate(df.columns):
        cell = table[(i+1, j)]
        if j > 0:  # Skip coloring the model name column
            cell.set_facecolor(apply_color_scale(df.iloc[i][j], col))
            
            # Format numbers to proper precision
            if col in ['Accuracy (%)', 'Precision (%)']:
                cell.get_text().set_text(f"{df.iloc[i][j]:.2f}")
            elif col == 'False Positive Rate':
                cell.get_text().set_text(f"{df.iloc[i][j]:.4f}")
            elif col == 'Avg Time (s)':
                cell.get_text().set_text(f"{df.iloc[i][j]:.4f}")
            elif col == 'Speed (faces/sec)':
                cell.get_text().set_text(f"{df.iloc[i][j]:.2f}")
        else:
            cell.set_facecolor('#f8f9fa')  # Light grey for model names

# Add a title and footnote
plt.title('Face Detection Models - Performance Metrics', pad=20, fontsize=16, fontweight='bold')
plt.figtext(0.5, 0.01, 'Color intensity indicates relative performance (darker is better)', 
           ha='center', fontsize=10, style='italic')

# Save the figure
plt.tight_layout()
plt.savefig('/home/samsepi0l/Project/FaceRecognition/face/papers/Face(Detection,Recognition)/Face_detection/enhanced_metrics_table.png', dpi=300, bbox_inches='tight')

print("Enhanced table image saved successfully!")
