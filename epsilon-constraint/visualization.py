"""
Visualization Module for Pareto Frontier and Results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


def plot_pareto_frontier(solutions: List[Dict], save_path: str = None):
    """
    Plot Pareto frontier in objective space

    Args:
        solutions: List of Pareto-optimal solutions
        save_path: Path to save figure (optional)
    """
    if not solutions:
        print("No solutions to plot")
        return

    costs = [sol['cost'] for sol in solutions]
    risks = [sol['risk'] for sol in solutions]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Pareto frontier
    ax.scatter(costs, risks, c='red', s=100, marker='o',
               label='Pareto-Optimal Solutions', zorder=3)
    ax.plot(costs, risks, 'r--', alpha=0.5, linewidth=1.5)

    # Annotate solutions
    for i, sol in enumerate(solutions):
        actions_str = ''.join([str(sol['actions'][a])
                               for a in sorted(sol['actions'].keys())])
        ax.annotate(actions_str, (sol['cost'], sol['risk']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

    ax.set_xlabel('Total Implementation Cost ($)', fontsize=12)
    ax.set_ylabel('Worst-Case Accident Risk P(N₁₀=H)', fontsize=12)
    ax.set_title('Pareto Frontier: Cost vs.  Safety Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Pareto frontier plot saved to {save_path}")

    plt.show()


def plot_action_heatmap(solutions: List[Dict], save_path: str = None):
    """
    Plot heatmap of action selection across Pareto solutions

    Args:
        solutions: List of Pareto-optimal solutions
        save_path:  Path to save figure (optional)
    """
    n_solutions = len(solutions)
    n_actions = 5

    action_matrix = np.zeros((n_solutions, n_actions))

    for i, sol in enumerate(solutions):
        for a in range(1, n_actions + 1):
            action_matrix[i, a - 1] = sol['actions'][a]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(action_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(n_solutions))
    ax.set_xticklabels([f"Sol {i + 1}" for i in range(n_solutions)], rotation=45)
    ax.set_yticks(range(n_actions))
    ax.set_yticklabels([f"Action {i + 1}" for i in range(n_actions)])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selected (1) / Not Selected (0)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(n_solutions):
        for j in range(n_actions):
            text = ax.text(i, j, int(action_matrix[i, j]),
                           ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Action Selection Across Pareto Solutions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Action heatmap saved to {save_path}")

    plt.show()


def export_results_table(solutions: List[Dict], save_path: str):
    """
    Export Pareto solutions to CSV file

    Args:
        solutions: List of solutions
        save_path: Path to save CSV
    """
    import pandas as pd

    data = []
    for i, sol in enumerate(solutions, 1):
        row = {
            'Solution_ID': i,
            'Cost': sol['cost'],
            'Risk': sol['risk'],
            'Risk_Reduction': None,  # Computed separately
            'Solve_Time': sol.get('solve_time', None)
        }
        for a in range(1, 6):
            row[f'Action_{a}'] = sol['actions'][a]
        data.append(row)

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✓ Results table saved to {save_path}")