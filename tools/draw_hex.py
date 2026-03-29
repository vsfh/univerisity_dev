import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import squarify
from matplotlib.colors import LinearSegmentedColormap
import json

def distance_visualize():
    # --- 1. Load distances.json ---
    with open("/data/feihong/ckpt/distances.json", "r") as f:
        data = json.load(f)
    
    # --- 2. Build n*n matrix ---
    keys = sorted(data.keys())
    n = len(keys)
    matrix = np.zeros((n, n))
    
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if k2 in data[k1]:
                matrix[i, j] = data[k1][k2]
            elif k1 in data[k2]:
                matrix[i, j] = data[k2][k1]
            else:
                matrix[i, j] = 0
    
    # --- 3. Create custom colormap (light -> dark) ---
    colors = ["#F9F8F5", "#B89386"]
    cmap = LinearSegmentedColormap.from_list("distance_cmap", colors)
    
    # --- 4. Visualize ---
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar(label='Distance')
    plt.xlabel("Index")
    plt.ylabel("Index")
    
    # --- 5. Save ---
    plt.savefig("runs/distance_matrix.png", dpi=150, bbox_inches='tight')
    print("Distance matrix saved to runs/distance_matrix.png")
    plt.close()

def squ():
    # --- 1. Your Data ---
    # Replace these lists with your actual categories, counts, and hex colors
    categories = ['Europe', 'North America', 'Asia', 'South America', 'Oceania', 'Africa']
    counts = [88, 86, 29, 3, 27, 2]

    # Colors matching the style in your reference image
    # You can use standard hex codes here
    colors = [
        "#F9F8F5", "#F2EFE9", "#E5E0D5", "#D1C7B7", "#B89386", "#4A453F"
    ]

    # --- 2. Format the Labels ---
    # This creates the "Name \n Number" format seen in the image
    # The {count:,} adds commas to large numbers (e.g., 31643 -> 31,643)
    labels = [f"{cat}\n{count:,}" for cat, count in zip(categories, counts)]

    # --- 3. Figure Setup ---
    # Set the font to match LaTeX/Academic styles
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(10, 6)) # Adjust width/height ratio as needed

    # --- 4. Draw the Treemap ---
    squarify.plot(
        sizes=counts, 
        label=labels, 
        color=colors, 
        alpha=0.95,          # Slight transparency for a softer look
        edgecolor="white",   # The clean white borders between boxes
        linewidth=2.5,       # Thickness of the white borders
        text_kwargs={
            'fontsize': 12, 
            'weight': 'bold', 
            'color': 'white' # White text contrasts best with colored blocks
        }
    )

    # --- 5. Academic Formatting ---
    plt.axis('off') # Treemaps do not need x/y axes
    plt.title("Regional Distribution of Target Landmarks", fontsize=16, fontweight='bold', pad=20)

    # --- 6. Save and Display ---
    # Save as a vector PDF for your ECCV LaTeX submission
    plt.savefig('runs/Continent_Treemap.png', dpi=300, bbox_inches='tight')

    print("Treemap generated and saved successfully!")
    plt.show()
def hex():
    # --- 1. Simulate Data for 5 Experiments & 5 Metrics with Different Units ---
    experiments = ['600', '540', '480', '420', '360', '300']
    metrics_data = {
        'Acc@25 (%)': [57.2, 60.5, 69.7, 81.2, 95.5, 99.5],   # Higher is better
        'Acc@50 (%)': [14.5, 21.2, 30.0, 37.2, 27.2, 35.2],        # Higher is better
        'mIoU': [29.4, 33.1, 38.5, 43.1, 43.5, 46.8],      # Lower is better
        'Center Dist': [84.4, 85.9, 82.4, 79.5, 69.7, 50.9], # Lower is better
    }

    rows = []
    for exp_idx, exp_name in enumerate(experiments):
        for metric_name, values in metrics_data.items():
            rows.append({
                'Experiment': exp_name,
                'Metric': metric_name,
                'Value': values[exp_idx]
            })

    df = pd.DataFrame(rows)

    # --- 2. Figure Settings for ECCV ---
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.family'] = 'serif'

    # --- 3. Create the Facet Grid ---
    # sharey=False is CRITICAL so each metric has its own unit scale
    g = sns.FacetGrid(df, col="Metric", col_wrap=4, sharey=False, height=3.5, aspect=0.9)


    # --- 4. Choose Your Plot Style ---
    custom_colors = ["#F9F8F5", "#F2EFE9", "#E5E0D5", "#D1C7B7", "#B89386", "#4A453F"]
    # OPTION A: Bar Chart (直方图)
    g.map(sns.barplot, "Experiment", "Value", order=experiments, palette=custom_colors, edgecolor=".2")

    # OPTION B: Point Plot (折线图风格) - Uncomment below to use point plot instead
    # g.map(sns.pointplot, "Experiment", "Value", order=experiments, color="#c0392b", markers="o", scale=1.2)


    # --- 5. Aesthetic Refinements ---
    g.set_titles("{col_name}", fontweight='bold', size=12)

    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    g.set_axis_labels("", "Value")
    plt.tight_layout()

    # --- 6. Save Output (UPDATED) ---
    # Save as PNG for easy viewing and sharing
    plt.savefig('WildMatch_Experimental_Results.png', dpi=300, bbox_inches='tight')

    print("Figure saved successfully as both PDF and PNG.")
    plt.show()


# squ()
distance_visualize()