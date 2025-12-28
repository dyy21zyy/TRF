"""
ε-Constraint Optimization Module
Implements the bi-objective optimization using Gurobi
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from bayesian_network import BayesianNetwork


class EpsilonConstraintOptimizer:
    """
    ε-Constraint method for Pareto frontier generation
    """

    def __init__(self, config_path: str, bn: BayesianNetwork):
        """
        Initialize optimizer

        Args:
            config_path: Path to configuration YAML file
            bn: BayesianNetwork instance
        """

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.bn = bn
        self.params = bn.params

        # Extract configuration
        self.solver_config = self.config['solver']['parameters']
        self.n_epsilon = self.config['algorithm']['epsilon_grid_size']
        self.warm_start = self.config['algorithm']['warm_start']

        # Problem dimensions
        self.n_actions = self.config['problem']['num_actions']
        self.n_scenarios = self.config['problem']['num_scenarios']
        self.scenarios = ['omega1', 'omega2', 'omega3']
        self.states = self.config['problem']['states']

        # Action costs
        self.costs = {int(k): v for k, v in self.params['action_costs'].items()}

        # Storage for Pareto solutions
        self.pareto_frontier = []

    def determine_epsilon_range(self) -> Tuple[float, float]:
        """
        Determine valid range [epsilon_min, epsilon_max]

        Returns:
            Tuple (epsilon_min, epsilon_max)
        """
        print("Phase I: Determining ε range...")

        # Compute epsilon_max (do-nothing baseline)
        actions_none = {i: 0 for i in range(1, self.n_actions + 1)}
        epsilon_max = max(
            self.bn.compute_risk(actions_none, scenario)
            for scenario in self.scenarios
        )

        print(f"  ε_max (baseline risk): {epsilon_max:.4f}")

        # Compute epsilon_min (all actions deployed)
        actions_all = {i: 1 for i in range(1, self.n_actions + 1)}
        epsilon_min = max(
            self.bn.compute_risk(actions_all, scenario)
            for scenario in self.scenarios
        )

        print(f"  ε_min (minimum achievable risk): {epsilon_min:.4f}")

        if epsilon_min >= epsilon_max:
            raise ValueError("Invalid ε range: interventions do not reduce risk!")

        return epsilon_min, epsilon_max

    def generate_epsilon_grid(
            self,
            epsilon_min: float,
            epsilon_max: float
    ) -> np.ndarray:
        """
        Generate uniformly spaced epsilon values

        Args:
            epsilon_min: Lower bound
            epsilon_max: Upper bound

        Returns:
            Array of epsilon values
        """
        return np.linspace(epsilon_min, epsilon_max, self.n_epsilon)

    def build_gurobi_model(
            self,
            epsilon: float,
            warm_start_actions: Optional[Dict[int, int]] = None
    ) -> Tuple[gp.Model, Dict]:
        """
        Build Gurobi MINLP model for subproblem P(ε)

        Args:
            epsilon: Risk tolerance threshold
            warm_start_actions:  Previous optimal actions for warm-start

        Returns:
            Tuple (Gurobi model, variable dictionary)
        """
        model = gp.Model(f"EpsilonConstraint_eps={epsilon:.4f}")

        # Configure solver parameters
        for param, value in self.solver_config.items():
            model.setParam(param, value)

        # Decision variables:  Binary action selection
        x = {}
        for a in range(1, self.n_actions + 1):
            x[a] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}")
            if warm_start_actions is not None:
                x[a].Start = warm_start_actions[a]

        # Auxiliary variable:  Worst-case risk
        Z2 = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Z2")

        # Marginal probability variables:  μ_{i,k}(ω)
        mu = {}
        for node in self.bn.all_nodes:
            for state in self.states:
                for scenario in self.scenarios:
                    var_name = f"mu_{node}_{state}_{scenario}"
                    mu[(node, state, scenario)] = model.addVar(
                        lb=0.0, ub=1.0,
                        vtype=GRB.CONTINUOUS,
                        name=var_name
                    )

        model.update()

        # Objective: Minimize total cost
        obj = gp.quicksum(self.costs[a] * x[a] for a in range(1, self.n_actions + 1))
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraint: Z2 ≤ ε (risk tolerance)
        model.addConstr(Z2 <= epsilon, name="epsilon_constraint")

        # Constraint: Z2 ≥ μ_{10,H}(ω) for all scenarios (worst-case)
        for scenario in self.scenarios:
            model.addConstr(
                Z2 >= mu[(10, 'H', scenario)],
                name=f"worstcase_{scenario}"
            )

        # Bayesian Network Inference Constraints
        self._add_bn_constraints(model, x, mu)

        var_dict = {'x': x, 'Z2': Z2, 'mu': mu}

        return model, var_dict

    def _add_bn_constraints(
            self,
            model: gp.Model,
            x: Dict[int, gp.Var],
            mu: Dict[Tuple, gp.Var]
    ):
        """
        Add Bayesian Network inference constraints

        Args:
            model: Gurobi model
            x: Action variables
            mu: Marginal probability variables
        """
        for scenario in self.scenarios:
            # Constraints for root nodes (Layer 1)
            for node in self.bn.root_nodes:
                self._add_root_node_constraints(model, x, mu, node, scenario)

            # Constraints for intermediate nodes (Layers 2-3)
            for node in self.bn.psych_nodes + self.bn.outcome_nodes:
                self._add_child_node_constraints(model, x, mu, node, scenario)

    def _add_root_node_constraints(
            self,
            model: gp.Model,
            x: Dict[int, gp.Var],
            mu: Dict[Tuple, gp.Var],
            node: int,
            scenario: str
    ):
        """
        Add constraints for root node probability adjustment
        """
        # Get baseline probabilities
        baseline = self.bn.baseline_probs[str(node)]

        # Build action effect expression
        action_effect_H = gp.LinExpr()
        for action_id in range(1, self.n_actions + 1):
            action_str = str(action_id)
            if action_str in self.params['action_efficacy']['physical']:
                action_data = self.params['action_efficacy']['physical'][action_str]
                if action_data['node'] == node:
                    action_effect_H += action_data['Gamma_H'] * x[action_id]

        # Get scenario adjustment
        scenario_data = self.params['scenario_adjustments'].get(scenario, {})
        theta_H = scenario_data.get(str(node), {}).get('H', 0.0)
        theta_M = scenario_data.get(str(node), {}).get('M', 0.0)

        # Conservatism margin
        delta_H = self.params['robustness_margin']['delta_H']

        # Intermediate adjusted values
        pi_tilde_H = baseline['H'] + action_effect_H + theta_H
        pi_tilde_M = baseline['M'] + theta_M

        # Robust adjustment
        pi_tilde_robust_H = pi_tilde_H + delta_H

        # ========== 关键修改：使用辅助变量代替 min/max ==========

        # 创建辅助变量
        pi_star_H = model.addVar(lb=0.0, ub=1.0, name=f"pi_star_{node}_H_{scenario}")
        pi_star_M = model.addVar(lb=0.0, ub=1.0, name=f"pi_star_{node}_M_{scenario}")

        # 约束：pi_star_H = clip(pi_tilde_robust_H, 0, 1)
        # 等价于：
        #   pi_star_H >= 0
        #   pi_star_H <= 1
        #   pi_star_H >= pi_tilde_robust_H  (如果 < 0)
        #   pi_star_H <= pi_tilde_robust_H  (如果在 [0,1])
        #   pi_star_H = 1                    (如果 > 1)

        # 使用 Gurobi 的 General Constraints
        model.addConstr(pi_star_H >= pi_tilde_robust_H, name=f"clip_lower_H_{node}_{scenario}")
        model.addConstr(pi_star_H <= 1.0, name=f"clip_upper_H_{node}_{scenario}")
        model.addConstr(pi_star_H >= 0.0, name=f"clip_nonneg_H_{node}_{scenario}")

        # 同样处理 M 状态
        model.addConstr(pi_star_M >= pi_tilde_M, name=f"clip_lower_M_{node}_{scenario}")
        model.addConstr(pi_star_M <= 1.0, name=f"clip_upper_M_{node}_{scenario}")
        model.addConstr(pi_star_M >= 0.0, name=f"clip_nonneg_M_{node}_{scenario}")

        # 计算 S = pi_star_H + pi_star_M
        S = model.addVar(lb=0.0, ub=2.0, name=f"S_{node}_{scenario}")
        model.addConstr(S == pi_star_H + pi_star_M, name=f"sum_S_{node}_{scenario}")

        # ========== 归一化处理：使用指示约束 ==========

        # 引入二元变量指示是否饱和
        z = model.addVar(vtype=GRB.BINARY, name=f"z_{node}_{scenario}")

        # z=1 表示 S <= 1 (Case 1)
        # z=0 表示 S > 1  (Case 2)

        M_big = 10.
        0  # Big-M 常数

        model.addConstr(S <= 1 + M_big * (1 - z), name=f"case1_cond_{node}_{scenario}")
        model.addConstr(S >= 1 + 1e-4 - M_big * z, name=f"case2_cond_{node}_{scenario}")

        # Case 1: S <= 1
        # mu_H = pi_star_H
        # mu_M = pi_star_M
        # mu_L = 1 - S

        model.addConstr(
            mu[(node, 'H', scenario)] <= pi_star_H + M_big * (1 - z),
            name=f"case1_H_upper_{node}_{scenario}"
        )
        model.addConstr(
            mu[(node, 'H', scenario)] >= pi_star_H - M_big * (1 - z),
            name=f"case1_H_lower_{node}_{scenario}"
        )

        model.addConstr(
            mu[(node, 'M', scenario)] <= pi_star_M + M_big * (1 - z),
            name=f"case1_M_upper_{node}_{scenario}"
        )
        model.addConstr(
            mu[(node, 'M', scenario)] >= pi_star_M - M_big * (1 - z),
            name=f"case1_M_lower_{node}_{scenario}"
        )

        model.addConstr(
            mu[(node, 'L', scenario)] <= (1 - S) + M_big * (1 - z),
            name=f"case1_L_upper_{node}_{scenario}"
        )
        model.addConstr(
            mu[(node, 'L', scenario)] >= (1 - S) - M_big * (1 - z),
            name=f"case1_L_lower_{node}_{scenario}"
        )

        # Case 2: S > 1
        # mu_H = pi_star_H / S
        # mu_M = pi_star_M / S
        # mu_L = 0

        # 由于除法是非线性的，引入辅助变量
        inv_S = model.addVar(lb=0.0, ub=10.0, name=f"inv_S_{node}_{scenario}")

        # inv_S * S = 1 (仅在 z=0 时生效)
        # 这是一个二次约束，需要 NonConvex=2
        model.addConstr(
            inv_S * S >= 1 - M_big * z,
            name=f"case2_inv_lower_{node}_{scenario}"
        )
        model.addConstr(
            inv_S * S <= 1 + M_big * z,
            name=f"case2_inv_upper_{node}_{scenario}"
        )

        model.addConstr(
            mu[(node, 'H', scenario)] <= pi_star_H * inv_S + M_big * z,
            name=f"case2_H_upper_{node}_{scenario}"
        )
        model.addConstr(
            mu[(node, 'H', scenario)] >= pi_star_H * inv_S - M_big * z,
            name=f"case2_H_lower_{node}_{scenario}"
        )

        model.addConstr(
            mu[(node, 'M', scenario)] <= pi_star_M * inv_S + M_big * z,
            name=f"case2_M_upper_{node}_{scenario}"
        )
        model.addConstr(
            mu[(node, 'M', scenario)] >= pi_star_M * inv_S - M_big * z,
            name=f"case2_M_lower_{node}_{scenario}"
        )

        model.addConstr(
            mu[(node, 'L', scenario)] <= M_big * z,
            name=f"case2_L_{node}_{scenario}"
        )

        # 归一化约束
        model.addConstr(
            mu[(node, 'L', scenario)] +
            mu[(node, 'M', scenario)] +
            mu[(node, 'H', scenario)] == 1,
            name=f"norm_{node}_{scenario}"
        )

    def _add_child_node_constraints(
            self,
            model: gp.Model,
            x: Dict[int, gp.Var],
            mu: Dict[Tuple, gp.Var],
            node: int,
            scenario: str
    ):
        """
        Add constraints for child node marginal probability computation

        使用递归二次化处理多父节点乘积
        """
        parents = self.bn.parents[str(node)]
        parent_configs = self.bn.get_parent_configs(node)

        for state in self.states:
            # Sum over all parent configurations
            marginal_expr = 0

            for config in parent_configs:
                if not config:  # 根节点，跳过
                    continue

                # 获取调整后的条件概率（预计算，不含变量）
                # 这里简化处理：用固定值
                actions_dict = {a: 0 for a in range(1, self.n_actions + 1)}
                cond_prob = self.bn.adjust_probabilities(node, config, actions_dict, scenario)

                # ========== 关键修改：处理父节点乘积 ==========

                if len(parents) == 1:
                    # 单父节点：直接相乘（线性）
                    parent_product = mu[(parents[0], config[0], scenario)]

                elif len(parents) == 2:
                    # 双父节点：直接相乘（二次，Gurobi支持）
                    parent1_var = mu[(parents[0], config[0], scenario)]
                    parent2_var = mu[(parents[1], config[1], scenario)]
                    parent_product = parent1_var * parent2_var

                elif len(parents) == 3:
                    # 三父节点：递归二次化
                    # 步骤1: 先计算 aux = mu[p1] * mu[p2]
                    # 步骤2: 再计算 product = aux * mu[p3]

                    parent1_var = mu[(parents[0], config[0], scenario)]
                    parent2_var = mu[(parents[1], config[1], scenario)]
                    parent3_var = mu[(parents[2], config[2], scenario)]

                    # 创建辅助变量
                    aux_name = f"aux_{node}_{config}_{scenario}"
                    aux_var = model.addVar(lb=0.0, ub=1.0, name=aux_name)

                    # 约束：aux = parent1 * parent2
                    model.addConstr(aux_var == parent1_var * parent2_var,
                                    name=f"bilinear_12_{node}_{config}_{scenario}")

                    # 最终乘积：product = aux * parent3
                    parent_product = aux_var * parent3_var

                else:
                    # 更多父节点：递归构造
                    raise NotImplementedError(f"节点 {node} 有 {len(parents)} 个父节点，暂不支持")

                # 累加到边际概率
                marginal_expr += cond_prob[state] * parent_product

            # 设置边际概率约束
            model.addConstr(
                mu[(node, state, scenario)] == marginal_expr,
                name=f"marginal_{node}_{state}_{scenario}"
            )

        # 归一化约束
        model.addConstr(
            sum(mu[(node, s, scenario)] for s in self.states) == 1,
            name=f"norm_{node}_{scenario}"
        )

    def solve_subproblem(
            self,
            epsilon: float,
            warm_start_actions: Optional[Dict[int, int]] = None
    ) -> Optional[Dict]:
        """
        Solve single ε-constrained subproblem

        Args:
            epsilon: Risk tolerance threshold
            warm_start_actions: Previous solution for warm-start

        Returns:
            Solution dictionary or None if infeasible
        """
        print(f"\n  Solving P(ε={epsilon:.4f}).. .", end=" ")

        model, var_dict = self.build_gurobi_model(epsilon, warm_start_actions)

        try:
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                # Extract solution
                x_sol = {a: int(var_dict['x'][a].X + 0.5)
                         for a in range(1, self.n_actions + 1)}
                Z1_sol = model.ObjVal
                Z2_sol = var_dict['Z2'].X

                print(f"✓ Optimal | Z1={Z1_sol:.0f} | Z2={Z2_sol:.4f}")

                return {
                    'epsilon': epsilon,
                    'actions': x_sol,
                    'cost': Z1_sol,
                    'risk': Z2_sol,
                    'solve_time': model.Runtime
                }

            elif model.Status == GRB.INFEASIBLE:
                print("✗ Infeasible")
                return None

            else:
                print(f"?  Status={model.Status}")
                return None

        except gp.GurobiError as e:
            print(f"✗ Gurobi Error: {e}")
            return None

    def generate_pareto_frontier(self) -> List[Dict]:
        """
        Generate Pareto frontier using ε-constraint method

        Returns:
            List of Pareto-optimal solutions
        """
        print("=" * 70)
        print("ε-CONSTRAINT METHOD FOR PARETO FRONTIER GENERATION")
        print("=" * 70)

        # Phase I:  Determine ε range
        epsilon_min, epsilon_max = self.determine_epsilon_range()
        epsilon_grid = self.generate_epsilon_grid(epsilon_min, epsilon_max)

        print(f"\nPhase II: Solving {self.n_epsilon} subproblems...")
        print("-" * 70)

        # Phase II: Iterative subproblem solution
        previous_actions = None

        for k, epsilon in enumerate(epsilon_grid, 1):
            print(f"[{k}/{self.n_epsilon}]", end="")

            # Solve with optional warm-start
            if self.warm_start and previous_actions is not None:
                solution = self.solve_subproblem(epsilon, previous_actions)
            else:
                solution = self.solve_subproblem(epsilon)

            if solution is not None:
                self.pareto_frontier.append(solution)
                previous_actions = solution['actions']

        # Phase III: Non-dominated filtering
        print("\nPhase III: Applying non-dominated filter...")
        self.pareto_frontier = self._filter_dominated(self.pareto_frontier)

        print(f"  Final Pareto set size: {len(self.pareto_frontier)}")
        print("=" * 70)

        return self.pareto_frontier

    def _filter_dominated(self, solutions: List[Dict]) -> List[Dict]:
        """
        Remove dominated solutions from Pareto set

        Args:
            solutions: List of solution dictionaries

        Returns:
            Filtered list of non-dominated solutions
        """
        non_dominated = []

        for sol_i in solutions:
            is_dominated = False

            for sol_j in solutions:
                if sol_i == sol_j:
                    continue

                # Check if sol_j dominates sol_i
                if (sol_j['cost'] <= sol_i['cost'] and
                        sol_j['risk'] <= sol_i['risk'] and
                        (sol_j['cost'] < sol_i['cost'] or sol_j['risk'] < sol_i['risk'])):
                    is_dominated = True
                    break

            if not is_dominated:
                non_dominated.append(sol_i)

        return non_dominated