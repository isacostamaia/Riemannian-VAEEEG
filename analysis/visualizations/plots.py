import os
import sys
from pathlib import Path

# Automatically get the root of the project (one level up from this script)
project_root = Path(__file__).resolve().parents[1]

#Add root to sys.path so you can import rvae, datasets, etc.
sys.path.append(str(project_root))

#Change working directory to root (for file paths)
os.chdir(project_root)


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def color_code(true_label, correct_pred_id):
    if true_label == 0 and correct_pred_id == 1:
        return 0 #true positive
    if true_label == 1 and correct_pred_id == 1:
        return 1 #true negative
    if true_label == 0 and correct_pred_id == 0:
        return 2 #false positive
    if true_label == 1 and correct_pred_id == 0:
        return 3 #false negative

def plot_2d(x1, x2, true_labels, pred_labels=None, exp_path=None, name=None):
    figsize = 8
    plt.figure(figsize=(figsize, figsize))
    plt.title(name)
    if ((pred_labels is None) or len(np.unique(true_labels))>2):
        #obs: 2D plot with predictions only implemented for binary classification
        color_codes = true_labels
        for cc in np.unique(color_codes):
            idx = color_codes == cc
            plt.scatter(x1[idx], x2[idx], label=f"True label: {cc}")        
    else:
        correct_pred_idx = true_labels==pred_labels  #index of correctly predicted samples
        color_codes = [color_code(tl, c_i) for tl, c_i in zip(true_labels, correct_pred_idx)]
        colors = ['blue', 'red', 'darkcyan', 'coral']
        texts = ['TN (0 pred 0)', 'TP (1 pred 1)', 'FP (0 pred 1)', 'FN (1 pred 0)']
        for cc in np.unique(color_codes):
            idx = color_codes == cc
            plt.scatter(x1[idx], x2[idx], label=texts[cc], c=colors[cc])
    plt.legend()
    if exp_path:
        fname = name.replace(' ', '_')
        plt.savefig(os.path.join(exp_path, "{}.png".format(fname)))
    plt.show()
    plt.close()

def plot_3d(x1, x2, x3, true_labels, pred_labels=None, exp_path=None, name=None):
    fig = go.Figure()

    if (pred_labels is None) or len(np.unique(true_labels)) > 2:
        #obs: 3D plot with predictions only implemented for binary classification
        unique_labels = np.unique(true_labels)
        for i, label in enumerate(unique_labels):
            idx = true_labels == label

            fig.add_trace(go.Scatter3d(
                x=x1[idx],
                y=x2[idx],
                z=x3[idx],
                mode='markers',
                marker=dict(
                    size=5,
                    opacity=0.8,
                ),
                text=[f"True label: {label}" for _ in range(np.sum(idx))],
                name=f"True label: {label}",
                showlegend=True
            ))
    else:
        label_pairs = np.unique(list(zip(true_labels, pred_labels)), axis=0)
        colors = ['blue', 'darkcyan', 'coral', 'red']

        for i, (tl, pl) in enumerate(label_pairs):
            color=colors[i % len(colors)]
            idx = (true_labels == tl) & (pred_labels == pl)
            fig.add_trace(go.Scatter3d(
                x=x1[idx],
                y=x2[idx],
                z=x3[idx],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.8,
                ),
                text=[f"True label: {tl},<br>Pred label: {pl}" for _ in range(np.sum(idx))],
                name=f"True: {tl}, Pred: {pl}",
                showlegend=True
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        title="{}".format(name),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            itemsizing='constant'
        )
    )

    if exp_path is not None and name is not None:
        fname = name.replace(' ', '_')
        fig.write_html(os.path.join(exp_path, f"3D_{fname}.html"))
    fig.show()
