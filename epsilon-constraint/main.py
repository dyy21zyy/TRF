"""
Main Script for Running ε-Constraint Optimization
"""

import argparse
import os
from bayesian_network import BayesianNetwork
from optimizer import EpsilonConstraintOptimizer
from data_loader import load_parameters
from visualization import plot_pareto_frontier, plot_action_heatmap, export_results_table


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pedestrian Safety Optimization')
    parser.add_argument('--config', type=str, default='config.json',  # ← 改这里
                        help='Path to configuration file')
    parser.add_argument('--params', type=str, default='data/parameters.json',
                        help='Path to parameter file')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 70)
    print("PEDESTRIAN SAFETY OPTIMIZATION - ε-CONSTRAINT METHOD")
    print("=" * 70 + "\n")

    # Load parameters
    print("Loading parameters...")
    parameters = load_parameters(args.params)

    # Initialize Bayesian Network
    print("Initializing Bayesian Network...")
    bn = BayesianNetwork(parameters)
    print(f"  Nodes: {len(bn.all_nodes)} | Scenarios: 3 | Actions: 5\n")

    # Initialize Optimizer
    print("Initializing ε-Constraint Optimizer...")
    optimizer = EpsilonConstraintOptimizer(args.config, bn)
    print(f"  Grid size: {optimizer.n_epsilon}")
    print(f"  Warm-start:  {optimizer.warm_start}\n")

    # Generate Pareto frontier
    pareto_solutions = optimizer.generate_pareto_frontier()

    # Visualize results
    print("\nGenerating visualizations...")
    plot_pareto_frontier(
        pareto_solutions,
        save_path=os.path.join(args.output, 'pareto_frontier.png')
    )
    plot_action_heatmap(
        pareto_solutions,
        save_path=os.path.join(args.output, 'action_heatmap.png')
    )

    # Export results
    print("\nExporting results...")
    export_results_table(
        pareto_solutions,
        save_path=os.path.join(args.output, 'pareto_solutions.csv')
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    # Print summary
    print("Summary:")
    print(f"  Total Pareto solutions: {len(pareto_solutions)}")
    print(f"  Cost range: ${pareto_solutions[0]['cost']:.0f} - ${pareto_solutions[-1]['cost']:.0f}")
    print(f"  Risk range: {pareto_solutions[-1]['risk']:.4f} - {pareto_solutions[0]['risk']:.4f}")
    print(f"\nResults saved to: {args.output}\n")


if __name__ == "__main__":
    main()