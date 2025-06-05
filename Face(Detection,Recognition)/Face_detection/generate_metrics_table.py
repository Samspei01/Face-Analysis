import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

# Create the figure and table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create a colormap for better visualization (highlight better values)
colors = ['#e6f7ff', '#cceeff', '#b3e6ff', '#99ddff', '#80d4ff']  # Light blue shades
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

# Add a title
plt.title('Face Detection Models - Performance Metrics', pad=20, fontsize=16, fontweight='bold')

# Save the figure
plt.tight_layout()
plt.savefig('/home/samsepi0l/Project/FaceRecognition/face/papers/Face(Detection,Recognition)/Face_detection/updated_metrics_table.png', dpi=300, bbox_inches='tight')

print("Table image saved successfully!")
