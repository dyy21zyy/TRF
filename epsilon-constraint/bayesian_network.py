"""
Bayesian Network Inference Module
Implements forward propagation with probability adjustment and normalization
"""

import numpy as np
from typing import Dict, List, Tuple
import itertools


class BayesianNetwork:
    """
    Bayesian Network for pedestrian safety risk assessment
    """

    def __init__(self, parameters: Dict):
        """
        Initialize Bayesian Network

        Args:
            parameters: Dictionary containing all model parameters
        """
        self.params = parameters
        self.nodes = parameters['nodes']
        self.parents = parameters['parents']
        self.baseline_probs = parameters['baseline_probabilities']
        self.states = ['L', 'M', 'H']

        # Categorize nodes
        self.root_nodes = self.nodes['root_nodes']
        self.psych_nodes = self.nodes['psych_nodes']
        self.outcome_nodes = self.nodes['outcome_nodes']
        self.all_nodes = self.root_nodes + self.psych_nodes + self.outcome_nodes

    def get_parent_configs(self, node: int) -> List[str]:
        """
        Generate all possible parent state configurations for a node

        Args:
            node: Node ID

        Returns:
            List of parent configuration strings (e.g., ['LL', 'LM', 'LH', ... ])
        """
        parents = self.parents[str(node)]

        if not parents:
            return ['']  # Empty string for root nodes

        # Generate all combinations of parent states
        parent_state_combinations = list(itertools.product(self.states, repeat=len(parents)))
        configs = [''.join(combo) for combo in parent_state_combinations]

        return configs

    def adjust_probabilities(
            self,
            node: int,
            parent_config: str,
            actions: Dict[int, int],
            scenario: str
    ) -> Dict[str, float]:
        """
        Adjust baseline probabilities based on actions and scenario

        Args:
            node: Node ID
            parent_config: Parent state configuration (empty for root nodes)
            actions: Dictionary {action_id: 0/1} indicating selected actions
            scenario: Scenario name ('omega1', 'omega2', 'omega3')

        Returns:
            Dictionary of adjusted probabilities {'L': p_L, 'M': p_M, 'H': p_H}
        """
        # Step 1: Get baseline probabilities
        baseline = self._get_baseline(node, parent_config)

        # Step 2: Intermediate adjustment (actions + scenario)
        pi_tilde = {}

        # High state
        pi_tilde['H'] = baseline['H']
        pi_tilde['H'] += self._apply_action_efficacy(node, parent_config, actions, 'H')
        pi_tilde['H'] += self._apply_scenario_adjustment(node, parent_config, scenario, 'H')

        # Medium state
        pi_tilde['M'] = baseline['M']
        pi_tilde['M'] += self._apply_scenario_adjustment(node, parent_config, scenario, 'M')

        # Step 3: Robust adjustment (add conservatism margin to High state)
        delta_H = self.params['robustness_margin']['delta_H']
        pi_tilde_robust_H = pi_tilde['H'] + delta_H

        # Step 4: Boundary enforcement (truncation)
        pi_star = {}
        pi_star['H'] = np.clip(pi_tilde_robust_H, 0, 1)
        pi_star['M'] = np.clip(pi_tilde['M'], 0, 1)

        # Step 5: Normalization and residual allocation
        S = pi_star['H'] + pi_star['M']

        if S <= 1:
            # Case 1: Feasible allocation
            pi_hat = {
                'H': pi_star['H'],
                'M': pi_star['M'],
                'L': 1 - S
            }
        else:
            # Case 2: Saturation (normalize proportionally)
            pi_hat = {
                'H': pi_star['H'] / S,
                'M': pi_star['M'] / S,
                'L': 0.0
            }

        # Ensure numerical stability
        total = sum(pi_hat.values())
        if abs(total - 1.0) > 1e-6:
            # Renormalize if needed
            pi_hat = {k: v / total for k, v in pi_hat.items()}

        return pi_hat

    def _get_baseline(self, node: int, parent_config: str) -> Dict[str, float]:
        """Get baseline probability for a node"""
        node_str = str(node)
        baseline_data = self.baseline_probs[node_str]

        if node in self.root_nodes:
            # Root nodes have direct probability
            return baseline_data
        else:
            # Child nodes have conditional probabilities
            return baseline_data['parent_config'][parent_config]

    def _apply_action_efficacy(
            self,
            node: int,
            parent_config: str,
            actions: Dict[int, int],
            state: str
    ) -> float:
        """Apply action efficacy coefficient"""
        if state != 'H':
            return 0.0  # Actions only affect High state

        action_params = self.params['action_efficacy']
        total_effect = 0.0

        # Physical interventions
        if node in [1, 3]:
            for action_id, is_selected in actions.items():
                if not is_selected:
                    continue
                action_str = str(action_id)
                if action_str in action_params['physical']:
                    action_data = action_params['physical'][action_str]
                    if action_data['node'] == node:
                        total_effect += action_data['Gamma_H']

        # Cognitive interventions
        if node in [5, 6, 7]:
            for action_id, is_selected in actions.items():
                if not is_selected:
                    continue
                action_str = str(action_id)
                if action_str in action_params['cognitive']:
                    action_data = action_params['cognitive'][action_str]
                    if action_data['node'] == node:
                        gamma_data = action_data['Gamma_H']
                        if isinstance(gamma_data, dict):
                            # Parent-state-dependent efficacy
                            total_effect += gamma_data.get(parent_config, 0.0)
                        else:
                            total_effect += gamma_data

        return total_effect

    def _apply_scenario_adjustment(
            self,
            node: int,
            parent_config: str,
            scenario: str,
            state: str
    ) -> float:
        """Apply scenario adjustment parameter"""
        scenario_data = self.params['scenario_adjustments'].get(scenario, {})
        node_str = str(node)

        if node_str not in scenario_data:
            return 0.0

        return scenario_data[node_str].get(state, 0.0)

    def forward_inference(
            self,
            actions: Dict[int, int],
            scenario: str
    ) -> Dict[int, Dict[str, float]]:
        """
        Perform forward Bayesian inference

        Args:
            actions: Dictionary {action_id: 0/1}
            scenario: Scenario name

        Returns:
            Dictionary {node_id: {'L': prob, 'M': prob, 'H': prob}}
        """
        # Store marginal probabilities
        marginals = {}

        # Phase 1: Root nodes (Layer 1)
        for node in self.root_nodes:
            marginals[node] = self.adjust_probabilities(node, '', actions, scenario)

        # Phase 2: Psychological nodes (Layer 2)
        for node in self.psych_nodes:
            marginals[node] = self._compute_marginal(node, marginals, actions, scenario)

        # Phase 3: Outcome nodes (Layer 3)
        for node in self.outcome_nodes:
            marginals[node] = self._compute_marginal(node, marginals, actions, scenario)

        return marginals

    def _compute_marginal(
            self,
            node: int,
            marginals: Dict[int, Dict[str, float]],
            actions: Dict[int, int],
            scenario: str
    ) -> Dict[str, float]:
        """
        Compute marginal probability for a child node using sum-product

        Args:
            node: Node ID
            marginals:  Already computed marginal probabilities
            actions: Action selection
            scenario: Scenario name

        Returns:
            Marginal probability distribution
        """
        parents = self.parents[str(node)]
        parent_configs = self.get_parent_configs(node)

        marginal = {'L': 0.0, 'M': 0.0, 'H': 0.0}

        for config in parent_configs:
            if not config:  # Should not happen for child nodes
                continue

            # Get adjusted conditional probability P(node=k | parents=config)
            cond_prob = self.adjust_probabilities(node, config, actions, scenario)

            # Compute joint probability of parent configuration
            joint_parent_prob = 1.0
            for i, parent_node in enumerate(parents):
                parent_state = config[i]
                joint_parent_prob *= marginals[parent_node][parent_state]

            # Accumulate marginal probability
            for state in self.states:
                marginal[state] += cond_prob[state] * joint_parent_prob

        # Ensure normalization (handle numerical errors)
        total = sum(marginal.values())
        if abs(total - 1.0) > 1e-6:
            marginal = {k: v / total for k, v in marginal.items()}

        return marginal

    def compute_risk(self, actions: Dict[int, int], scenario: str) -> float:
        """
        Compute P(N10 = H | actions, scenario)

        Args:
            actions: Action selection dictionary
            scenario: Scenario name

        Returns:
            Probability of high accident risk
        """
        marginals = self.forward_inference(actions, scenario)
        return marginals[10]['H']