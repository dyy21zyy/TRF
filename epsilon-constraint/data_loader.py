"""
Data Loading and Validation Module
"""

import json
from typing import Dict


def load_parameters(filepath: str) -> Dict:
    """
    Load model parameters from JSON file

    Args:
        filepath: Path to parameters. json

    Returns:
        Dictionary of parameters
    """
    with open(filepath, 'r') as f:
        params = json.load(f)

    # Validate parameters
    _validate_parameters(params)

    return params


def _validate_parameters(params: Dict):
    """
    Validate parameter structure and consistency

    Args:
        params: Parameter dictionary

    Raises:
        ValueError: If parameters are invalid
    """
    required_keys = [
        'nodes', 'parents', 'baseline_probabilities',
        'action_efficacy', 'scenario_adjustments',
        'robustness_margin', 'action_costs'
    ]

    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter key: {key}")

    # Validate probability distributions sum to 1
    for node_id, probs in params['baseline_probabilities'].items():
        if isinstance(probs, dict) and 'L' in probs:
            # Root node
            total = probs['L'] + probs['M'] + probs['H']
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Node {node_id} baseline probabilities do not sum to 1: {total}"
                )

    print("✓ Parameters validated successfully")