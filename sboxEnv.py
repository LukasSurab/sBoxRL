import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sbox_utils import (generate_sbox, sbox_logic, get_sb_props, evaluate_dspectrum_metrics, evaluate_lspectrum_metrics,
                        log_sbox_hex, dspectrum)
import datetime

class SBoxEnv(gym.Env):
    """
    Environment for designing an 8x8 S-box with configurable reward functions.
    When the reward_mode is "spectral_linear" or "spectral_linear_du", the environment
    calculates and stores linearity (the maximum Walsh coefficient) instead of nonlinearity.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps=10000, reward_mode="nonlinearity", init_random_ops=True,
                 random_ops_count=10, non_improvement_limit=100, reward_config=None,
                 allowed_gates=None, model_save_path=None):
        super(SBoxEnv, self).__init__()
        self.max_steps = max_steps
        self._max_episode_steps = max_steps
        self.current_step = 0
        self.reward_mode = reward_mode
        self.non_improvement_limit = non_improvement_limit

        self.best_du10_nonl96 = None
        self.best_du10_nonl96_mc = np.inf  # Lower multiplicative complexity is better.
        self.best_du10_nonl96_gate_count = np.inf  # Tiebreaker: lower gate count.
        self.best_du10_nonl96_operations = []

        # Best S-box with Differential Uniformity (DU)=10 and Nonlinearity=94.
        self.best_du10_nonl94 = None
        self.best_du10_nonl94_mc = np.inf
        self.best_du10_nonl94_gate_count = np.inf
        self.best_du10_nonl94_operations = []

        if allowed_gates is None:
            self.allowed_gates = ["XOR", "TOFFOLI", "NOT", "FREDKIN"]
        else:
            self.allowed_gates = allowed_gates

        # Flatten the original action space: [len(allowed_gates), 8, 8, 8]
        n_gates = len(self.allowed_gates)
        n_param = 8  # each parameter from 0 to 7
        self.total_actions = n_gates * (n_param ** 3)
        self.action_space = spaces.Discrete(self.total_actions)

        # Observation space remains as before.
        self.observation_space = spaces.Box(low=0, high=self.max_steps,
                                            shape=(len(self.allowed_gates),), dtype=np.int32)
        self.operation_counts = [0] * len(self.allowed_gates)
        self.logic_json = {"reverse_bits": False, "operations": []}

        # For modes using nonlinearity.
        self.prev_nonlinearity = None
        # For modes using linearity.
        self.prev_linearity = None
        self.prev_du = None
        self.prev_ds_metric = None
        self.prev_ls_metric = None

        self.episode_best_score = -np.inf
        self.episode_best_sbox = None
        self.episode_best_metrics = None
        self.episode_best_operations = []
        self.episode_best_ds_freq = None

        self.global_best_score = -np.inf
        self.global_best_sbox = None
        self.global_best_metrics = None
        self.global_best_operations = []
        self.global_best_ds_freq = None

        self.non_improvement_count = 0

        self.init_random_ops = init_random_ops
        self.random_ops_count = random_ops_count

        if reward_config is None:
            self.reward_config = {
                'w_nonl': 1.0,
                'w_du': 1.0,
                'w_gate': 0.1,
                'w_explore': 1.0,
                'w_no_improve': 0.05,
                'nonl_bonus_threshold': 98,
                'nonl_bonus_factor': 100,
                'du_bonus_threshold': 10,
                'du_bonus_factor': 100,
                'nonl_bonus_threshold2': 100,
                'nonl_bonus_factor2': 1000,
                'du_bonus_threshold2': 8,
                'du_bonus_factor2': 1000,
                'w_toffoli': 0.1,
                'w_fredkin': 0.2,
                'spectral_du_weight': 0.01,
                'spectral_nonl_weight': 0.01,
                "spectral_linear_weight": 0.01,
            }
        else:
            self.reward_config = reward_config

        self.reward_functions = {
            "nonlinearity": self.reward_nonlinearity,
            "du": self.reward_du,
            "combined": self.reward_combined,
            "complexity": self.reward_complexity,
            "spectral_combined": self.reward_spectral_combined,
            "spectral_du": self.reward_spectral_du,
            "spectral_nonl": self.reward_spectral_nonl,
            "spectral_linear": self.reward_spectral_linear,
            "spectral_linear_du": self.reward_spectral_linear_du,
        }

        if model_save_path is None:
            allowed_str = "-".join(self.allowed_gates)
            self.model_save_path = f"zopravaSpecpdatingSiLU_{self.reward_mode}_{allowed_str}_nonimp{self.non_improvement_limit}.zip"
        else:
            self.model_save_path = model_save_path

        self.file_prefix = f"zopravaSpecpdatingSiLU_{self.reward_mode}_{'-'.join(self.allowed_gates)}_nonimp{self.non_improvement_limit}"

        self.episode_history = []         # List of episode summaries
        self.progress_history = []        # List of per-step progress entries
        self.all_progress_histories = []  # List of progress histories for each episode

    def _decode_action(self, action_int):
        """Decode a single integer action into (gate_idx, target, control1, control2)."""
        n_param = 8
        n_cube = n_param ** 3  # 512
        gate_idx = action_int // n_cube
        remainder = action_int % n_cube
        target = remainder // (n_param ** 2)
        remainder2 = remainder % (n_param ** 2)
        control1 = remainder2 // n_param
        control2 = remainder2 % n_param
        return int(gate_idx), int(target), int(control1), int(control2)

    # ------------------ Reward Functions ------------------ #
    def reward_nonlinearity(self, nonl, prev_nonl, du, prev_du):
        base = nonl - prev_nonl
        bonus_nonl = 0
        if nonl > self.reward_config['nonl_bonus_threshold']:
            bonus_nonl += self.reward_config['nonl_bonus_factor'] * (nonl - self.reward_config['nonl_bonus_threshold'])
        if nonl > self.reward_config['nonl_bonus_threshold2']:
            bonus_nonl += self.reward_config['nonl_bonus_factor2'] * (nonl - self.reward_config['nonl_bonus_threshold2'])
        return base + bonus_nonl

    def reward_du(self, nonl, prev_nonl, du, prev_du):
        base = prev_du - du
        bonus_du = 0
        if du < self.reward_config['du_bonus_threshold']:
            bonus_du += self.reward_config['du_bonus_factor'] * (self.reward_config['du_bonus_threshold'] - du)
        if du < self.reward_config['du_bonus_threshold2']:
            bonus_du += self.reward_config['du_bonus_factor2'] * (self.reward_config['du_bonus_threshold2'] - du)
        return base + bonus_du

    def reward_combined(self, nonl, prev_nonl, du, prev_du):
        base = (nonl - prev_nonl) + (prev_du - du)
        bonus_nonl = 0
        bonus_du = 0
        if nonl > self.reward_config['nonl_bonus_threshold']:
            bonus_nonl += self.reward_config['nonl_bonus_factor'] * (nonl - self.reward_config['nonl_bonus_threshold'])
        if nonl > self.reward_config['nonl_bonus_threshold2']:
            bonus_nonl += self.reward_config['nonl_bonus_factor2'] * (nonl - self.reward_config['nonl_bonus_threshold2'])
        if du < self.reward_config['du_bonus_threshold']:
            bonus_du += self.reward_config['du_bonus_factor'] * (self.reward_config['du_bonus_threshold'] - du)
        if du < self.reward_config['du_bonus_threshold2']:
            bonus_du += self.reward_config['du_bonus_factor2'] * (self.reward_config['du_bonus_threshold2'] - du)
        return base + bonus_nonl + bonus_du

    def reward_complexity(self, nonl, prev_nonl, du, prev_du):
        base = (nonl - du)
        bonus_nonl = 0
        bonus_du = 0
        if nonl > self.reward_config['nonl_bonus_threshold']:
            bonus_nonl += self.reward_config['nonl_bonus_factor'] * (nonl - self.reward_config['nonl_bonus_threshold'])
        if nonl > self.reward_config['nonl_bonus_threshold2']:
            bonus_nonl += self.reward_config['nonl_bonus_factor2'] * (nonl - self.reward_config['nonl_bonus_threshold2'])
        if du < self.reward_config['du_bonus_threshold']:
            bonus_du += self.reward_config['du_bonus_factor'] * (self.reward_config['du_bonus_threshold'] - du)
        if du < self.reward_config['du_bonus_threshold2']:
            bonus_du += self.reward_config['du_bonus_factor2'] * (self.reward_config['du_bonus_threshold2'] - du)
        return base + bonus_nonl + bonus_du

    def reward_spectral_combined(self, nonl, prev_nonl, du, prev_du, ds_val, ls_val, non_improve_count):
        """
        Reward combining improvements in differential uniformity and nonlinearity (or linearity)
        with a modified bonus that grows more steeply from nonl=96 upward.

        Parameters:
          nonl, prev_nonl: current and previous nonlinearity (or linearity) values.
          du, prev_du: current and previous differential uniformity values.
          ds_val, ls_val: spectral values used in bonus adjustments.
          non_improve_count: consecutive non-improvement step count (penalty factor).

        Returns:
          reward: a float representing the computed reward.
        """
        # Explicitly cast metric values to Python floats:
        n_val = float(nonl)
        prev_n_val = float(prev_nonl)
        d_val = float(du)
        prev_d_val = float(prev_du)

        # Calculate improvements (deltas)
        nonl_delta = n_val - prev_n_val
        du_delta = prev_d_val - d_val  # (du decreases when improving)

        # --- Piecewise Logistic Multiplier for nonlinearity improvements ---
        if n_val < 96:
            # Softer logistic scaling for lower values:
            max_nonl_mult = 3.0  # maximum multiplier value in this region
            threshold1 = 93.0  # threshold for bonus growth
            k1 = 0.1  # logistic steepness for the first branch
            nonl_multiplier = 1 + (max_nonl_mult - 1) / (1 + np.exp(-k1 * (n_val - threshold1)))
        else:
            # Steeper logistic scaling for nonl >= 96.
            max_nonl_mult_2 = 5  # increased maximum multiplier in the high-performance region
            threshold2 = 96.0
            k2 = 0.5  # steeper slope for faster bonus increase
            nonl_multiplier = 1 + (max_nonl_mult_2 - 1) / (1 + np.exp(-k2 * (n_val - threshold2)))

        weighted_nonl_delta = nonl_delta * nonl_multiplier

        # --- Logistic scaling for differential uniformity (du) ---
        du_threshold = 10.0
        du_k = 0.05
        max_du_mult = 2.0
        du_multiplier = 1 + (max_du_mult - 1) / (1 + np.exp(-du_k * (du_threshold - d_val)))
        weighted_du_delta = du_delta * du_multiplier

        # Bonus terms as originally defined:
        bonus_nonl = 0
        bonus_du = 0
        if n_val > self.reward_config['nonl_bonus_threshold']:
            bonus_nonl += self.reward_config['nonl_bonus_factor'] * (n_val - self.reward_config['nonl_bonus_threshold'])
        if n_val > self.reward_config['nonl_bonus_threshold2']:
            bonus_nonl += self.reward_config['nonl_bonus_factor2'] * (
                    n_val - self.reward_config['nonl_bonus_threshold2'])
        if d_val < self.reward_config['du_bonus_threshold']:
            bonus_du += self.reward_config['du_bonus_factor'] * (self.reward_config['du_bonus_threshold'] - d_val)
        if d_val < self.reward_config['du_bonus_threshold2']:
            bonus_du += self.reward_config['du_bonus_factor2'] * (self.reward_config['du_bonus_threshold2'] - d_val)

        if n_val == prev_n_val:
            current_ls = evaluate_lspectrum_metrics(ls_val)[1]
            bonus_spectral_nonl = self.reward_config['spectral_nonl_weight'] * (self.prev_ls_metric - current_ls)
            self.prev_ls_metric = current_ls
        else:
            bonus_spectral_nonl = 0

        if d_val == prev_d_val:
            current_ds = evaluate_dspectrum_metrics(ds_val)[1]
            bonus_spectral_du = self.reward_config['spectral_du_weight'] * (self.prev_ds_metric - current_ds)
            self.prev_ds_metric = current_ds
        else:
            bonus_spectral_du = 0

        penalty = self.reward_config['w_no_improve'] * non_improve_count

        current_mc = sum(1 for op in self.logic_json["operations"] if op['type'] in ["TOFFOLI", "FREDKIN"])
        mc_penalty = self.reward_config.get("w_mc", 0.1) * current_mc

        reward = bonus_nonl + bonus_du + (
                weighted_nonl_delta + weighted_du_delta + bonus_spectral_du + bonus_spectral_nonl - penalty - mc_penalty) * 0.1
        return reward

    def reward_spectral_du(self, nonl, prev_nonl, du, prev_du, ds_val, ls_val, non_improve_count):
        base = prev_du - du
        bonus_du = 0
        bonus_spectral_du = 0
        if du < self.reward_config['du_bonus_threshold']:
            bonus_du += self.reward_config['du_bonus_factor'] * (self.reward_config['du_bonus_threshold'] - du)
        if du < self.reward_config['du_bonus_threshold2']:
            bonus_du += self.reward_config['du_bonus_factor2'] * (self.reward_config['du_bonus_threshold2'] - du)
        current_ds = evaluate_dspectrum_metrics(ds_val)[1]
        if base == 0:
            bonus_spectral_du = self.reward_config['spectral_du_weight'] * (self.prev_ds_metric - current_ds)
        self.prev_ds_metric = current_ds
        penalty = self.reward_config['w_no_improve'] * non_improve_count
        reward = (base + bonus_spectral_du - penalty) * 0.1 + bonus_du
        return reward


    def reward_spectral_nonl(self, nonl, prev_nonl, du, prev_du, ds_val, ls_val, non_improve_count):
        base = nonl - prev_nonl
        bonus_nonl = 0
        bonus_spectral_nonl = 0
        if nonl > self.reward_config['nonl_bonus_threshold']:
            bonus_nonl += self.reward_config['nonl_bonus_factor'] * (nonl - self.reward_config['nonl_bonus_threshold'])
        if nonl > self.reward_config['nonl_bonus_threshold2']:
            bonus_nonl += self.reward_config['nonl_bonus_factor2'] * (nonl - self.reward_config['nonl_bonus_threshold2'])
        current_ls = evaluate_lspectrum_metrics(ls_val)[1]
        if base == 0:
            bonus_spectral_nonl = self.reward_config['spectral_nonl_weight'] * (self.prev_ls_metric - current_ls)
        self.prev_ls_metric = current_ls
        penalty = self.reward_config['w_no_improve'] * non_improve_count
        return (base + bonus_nonl + bonus_spectral_nonl - penalty) * 0.1


    def reward_spectral_linear(self, nonl, prev_nonl, du, prev_du, ds_val, ls_val, non_improve_count):
        """
        Reward based solely on improvements in linearity.
        Linearity is measured as the maximum Walsh coefficient (lin = ls_val[-1][0]).
        A decrease in this value is rewarded.
        """
        current_linear = ls_val[-1][0]
        if self.prev_linearity is None:
            self.prev_linearity = current_linear
        base = self.prev_linearity - current_linear
        bonus = 0
        if base == 0:
            bonus = self.reward_config.get("spectral_linear_weight", 0.01) * (self.prev_linearity - current_linear)
        self.prev_linearity = current_linear
        penalty = self.reward_config['w_no_improve'] * non_improve_count
        reward = (base + bonus - penalty) * 0.1
        return reward

    def reward_spectral_linear_du(self, nonl, prev_nonl, du, prev_du, ds_val, ls_val, non_improve_count):
        """
        Reward combining improvements in differential uniformity and linearity.
        Differential uniformity component is computed as in spectral_du.
        Linearity improvement is measured as the reduction in the maximum Walsh coefficient.
        """
        base_du = prev_du - du
        bonus_du = 0
        if du < self.reward_config['du_bonus_threshold']:
            bonus_du += self.reward_config['du_bonus_factor'] * (self.reward_config['du_bonus_threshold'] - du)
        if du < self.reward_config['du_bonus_threshold2']:
            bonus_du += self.reward_config['du_bonus_factor2'] * (self.reward_config['du_bonus_threshold2'] - du)
        current_linear = ls_val[-1][0]
        if self.prev_linearity is None:
            self.prev_linearity = current_linear
        base_linear = self.prev_linearity - current_linear
        bonus_linear = 0
        if base_linear == 0:
            bonus_linear = self.reward_config.get("spectral_linear_weight", 0.01) * (self.prev_linearity - current_linear)
        self.prev_linearity = current_linear
        base_combined = base_du + base_linear
        bonus_combined = bonus_du + bonus_linear
        penalty = self.reward_config['w_no_improve'] * non_improve_count
        reward = (base_combined + bonus_combined - penalty) * 0.1
        return reward

    # ------------------ Environment Reset and Step ------------------ #
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.operation_counts = [0] * len(self.allowed_gates)
        self.logic_json["operations"] = []
        self.non_improvement_count = 0

        self.episode_best_score = -np.inf
        self.episode_best_sbox = None
        self.episode_best_metrics = None
        self.episode_best_operations = []
        self.episode_best_ds_freq = None
        self.progress_history = []

        if self.init_random_ops:
            for i in range(self.random_ops_count):
                gate_idx = np.random.randint(0, len(self.allowed_gates))
                gate_type = self.allowed_gates[gate_idx]
                if gate_type == "XOR":
                    target, control1 = np.random.choice(range(8), size=2, replace=False)
                    op = {'type': 'XOR', 'target': int(target), 'control1': int(control1)}
                elif gate_type == "TOFFOLI":
                    target, control1, control2 = np.random.choice(range(8), size=3, replace=False)
                    op = {'type': 'TOFFOLI', 'target': int(target), 'control1': int(control1), 'control2': int(control2)}
                elif gate_type == "NOT":
                    target = np.random.choice(range(8))
                    op = {'type': 'NOT', 'target': int(target)}
                elif gate_type == "FREDKIN":
                    control, target1, target2 = np.random.choice(range(8), size=3, replace=False)
                    op = {'type': 'FREDKIN', 'control': int(control), 'target1': int(target1), 'target2': int(target2)}
                self.logic_json["operations"].append(op)
                self.operation_counts[gate_idx] += 1

        sbox = generate_sbox(sbox_logic, self.logic_json)
        du, ds_val, ls_val, nonl, lin = get_sb_props(sbox)
        if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            self.prev_linearity = lin
            metric_value = lin
        else:
            self.prev_nonlinearity = nonl
            metric_value = nonl
        self.prev_du = du
        self.prev_ds_metric = evaluate_dspectrum_metrics(ds_val)[1]
        self.prev_ls_metric = evaluate_lspectrum_metrics(ls_val)[1]

        print("\n--- Environment Reset ---")
        print("Initial S-box (Hex):")
        print(log_sbox_hex(sbox))
        if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            print("Initial Metrics -> DU:", du, ", Linearity:", lin)
        else:
            print("Initial Metrics -> DU:", du, ", Nonlinearity:", nonl)
        worst_diff, worst_freq, avg_ds = evaluate_dspectrum_metrics(ds_val)
        worst_walsh, worst_walsh_freq, avg_ls = evaluate_lspectrum_metrics(ls_val)
        print("Initial Differential Spectrum -> Worst-case diff:", worst_diff,
              "Frequency:", worst_freq, "Avg Frequency:", avg_ds)
        print("Initial Linear Spectrum -> Worst Walsh:", worst_walsh,
              "Frequency:", worst_walsh_freq, "Avg Frequency:", avg_ls)
        print("Initial Gate Count:", len(self.logic_json["operations"]))

        if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            self.progress_history.append({
                'gate_count': len(self.logic_json["operations"]),
                'linearity': metric_value,
                'du': du
            })
        else:
            self.progress_history.append({
                'gate_count': len(self.logic_json["operations"]),
                'nonlinearity': metric_value,
                'du': du
            })

        return np.array(self.operation_counts, dtype=np.int32), {}

    def step(self, action):
        # Decode the action.
        gate_idx, target, control1, control2 = self._decode_action(action)
        self.current_step += 1
        gate_type = self.allowed_gates[gate_idx]

        if self.current_step == 1:
            # Never pick XOR as the first gate
            non_xor_gates = [g for g in self.allowed_gates if g != "XOR"]
            if non_xor_gates:
                # choose one at random (or you could pick a fixed one, e.g. TOFFOLI)
                gate_type = np.random.choice(non_xor_gates)
                gate_idx = self.allowed_gates.index(gate_type)

        if gate_type == "XOR":
            if target == control1:
                available = list(set(range(8)) - {target})
                control1 = np.random.choice(available)
            op = {'type': 'XOR', 'target': int(target), 'control1': int(control1)}
        elif gate_type == "TOFFOLI":
            if target == control1 or target == control2 or control1 == control2:
                target, control1, control2 = np.random.choice(range(8), size=3, replace=False)
            op = {'type': 'TOFFOLI', 'target': int(target), 'control1': int(control1), 'control2': int(control2)}
        elif gate_type == "NOT":
            op = {'type': 'NOT', 'target': int(target)}
        elif gate_type == "FREDKIN":
            if target == control1 or target == control2 or control1 == control2:
                control, t1, t2 = np.random.choice(range(8), size=3, replace=False)
                op = {'type': 'FREDKIN', 'control': int(control), 'target1': int(t1), 'target2': int(t2)}
            else:
                op = {'type': 'FREDKIN', 'control': int(target), 'target1': int(control1), 'target2': int(control2)}
        else:
            op = {'type': 'XOR', 'target': int(target), 'control1': int(control1)}

        self.operation_counts[gate_idx] += 1
        self.logic_json["operations"].append(op)

        new_gate_penalty = 0
        if op['type'] == 'TOFFOLI':
            new_gate_penalty = self.reward_config['w_toffoli']
        elif op['type'] == 'FREDKIN':
            new_gate_penalty = self.reward_config['w_fredkin']

        sbox = generate_sbox(sbox_logic, self.logic_json)
        du, ds_val, ls_val, nonl, lin = get_sb_props(sbox)

        current_gate_count = len(self.logic_json["operations"])
        # Compute multiplicative complexity (MC): count each TOFFOLI and FREDKIN gate.
        current_mc = sum(1 for op in self.logic_json["operations"] if op['type'] in ["TOFFOLI", "FREDKIN"])

        if du == 10 and nonl == 96:
            if (current_mc < self.best_du10_nonl96_mc) or \
                    (current_mc == self.best_du10_nonl96_mc and current_gate_count < self.best_du10_nonl96_gate_count):
                self.best_du10_nonl96 = sbox
                self.best_du10_nonl96_operations = self.logic_json["operations"].copy()
                self.best_du10_nonl96_mc = current_mc
                self.best_du10_nonl96_gate_count = current_gate_count
                print("Updated best S-box with DU=10 and Nonlinearity=96!")
                # Write the updated best S-box to a text file.
                best_filename_96 = f"best_Sbox_DU10_Nonl96_step.txt"
                with open(best_filename_96, "w") as f:
                    f.write("\n--- Best S-box with DU=10 and Nonlinearity=96 ---\n")
                    f.write(log_sbox_hex(sbox) + "\n")
                    f.write(f"Multiplicative Complexity: {current_mc}, Gate Count: {current_gate_count}\n")
                    f.write("Operations:\n")
                    for op in self.logic_json["operations"]:
                        f.write(str(op) + "\n")

        # For S-box with DU = 10 and Nonlinearity = 94.
        if du == 10 and nonl == 94:
            if (current_mc < self.best_du10_nonl94_mc) or \
                    (current_mc == self.best_du10_nonl94_mc and current_gate_count < self.best_du10_nonl94_gate_count):
                self.best_du10_nonl94 = sbox
                self.best_du10_nonl94_operations = self.logic_json["operations"].copy()
                self.best_du10_nonl94_mc = current_mc
                self.best_du10_nonl94_gate_count = current_gate_count
                print("Updated best S-box with DU=10 and Nonlinearity=94!")
                # Write the updated best S-box to a text file.
                best_filename_94 = f"best_Sbox_DU10_Nonl94.txt"
                with open(best_filename_94, "w") as f:
                    f.write("\n--- Best S-box with DU=10 and Nonlinearity=94 ---\n")
                    f.write(log_sbox_hex(sbox) + "\n")
                    f.write(f"Multiplicative Complexity: {current_mc}, Gate Count: {current_gate_count}\n")
                    f.write("Operations:\n")
                    for op in self.logic_json["operations"]:
                        f.write(str(op) + "\n")

        worst_diff, worst_freq, avg_ds = evaluate_dspectrum_metrics(ds_val)
        worst_walsh, worst_walsh_freq, avg_ls = evaluate_lspectrum_metrics(ls_val)
        print(f"DSpectrum: Worst-case diff (du): {worst_diff}, Frequency: {worst_freq}, Avg Frequency: {avg_ds}")
        print(f"LSpectrum: Worst Walsh: {worst_walsh}, Frequency: {worst_walsh_freq}, Avg Frequency: {avg_ls}")

        # --- Added for critical logging: Log S-box if linearity < 30 or du < 10 ---
        if nonl > 98 or du < 10:
            log_filename = f"critical_log_step_{self.current_step}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_filename, "w") as log_file:
                log_file.write(f"Critical S-box found at Step {self.current_step}\n")
                log_file.write(f"S-box (Hex): {log_sbox_hex(sbox)}\n")
                log_file.write(f"Metrics -> DU: {du}, Nonlinearity: {nonl}, Linearity: {lin}\n")
                log_file.write("Sequence of Applied Operations:\n")
                for op in self.logic_json['operations']:
                    log_file.write(str(op) + "\n")
        # --- End Added for critical logging ---

        # Choose the primary metric based on reward_mode.
        if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            metric = lin
        else:
            metric = nonl

        if self.reward_mode in ["nonlinearity", "spectral_nonl"]:
            score = nonl
        elif self.reward_mode in ["du", "spectral_du"]:
            score = -du
        elif self.reward_mode in ["combined", "spectral_combined"]:
            score = nonl - du
        elif self.reward_mode in ["complexity"]:
            score = (nonl - du)
        elif self.reward_mode in ["spectral_linear"]:
            score = -lin  # Lower linearity (i.e. lower maximum Walsh) is better.
        elif self.reward_mode in ["spectral_linear_du"]:
            score = (-lin - du)
        else:
            score = nonl

        current_gate_count = len(self.logic_json["operations"])

        # --- Composite Metric Update Block ---
        worst_diff, worst_freq, avg_ds = evaluate_dspectrum_metrics(ds_val)
        worst_walsh, worst_walsh_freq, avg_ls = evaluate_lspectrum_metrics(ls_val)
        # Compute multiplicative complexity (MC): each TOFFOLI and FREDKIN gate adds one.
        mc = sum(1 for op in self.logic_json["operations"] if op['type'] in ["TOFFOLI", "FREDKIN"])

        if self.reward_mode in ["nonlinearity", "spectral_nonl"]:
            composite_metrics = (nonl, -mc, -worst_walsh_freq,   -current_gate_count)
        elif self.reward_mode in ["du", "spectral_du"]:
            composite_metrics = (-du, -worst_freq, -mc,   -current_gate_count)
        elif self.reward_mode in ["combined", "spectral_combined"]:
            composite_metrics = ((nonl - du), -mc,   -current_gate_count)
        elif self.reward_mode in ["spectral_linear"]:
            composite_metrics = (-lin, -mc, -worst_walsh_freq,  -current_gate_count)
        elif self.reward_mode in ["spectral_linear_du"]:
            composite_metrics = ((-lin - du),-mc, -(worst_walsh_freq + worst_freq),   -current_gate_count)
        else:
            composite_metrics = (score, 0, -mc, -current_gate_count)

        new_score, new_mc, new_gate_count = composite_metrics

        if (self.episode_best_metrics is None) or (composite_metrics > self.episode_best_metrics):
            self.episode_best_metrics = composite_metrics
            self.episode_best_score = new_score
            self.episode_best_sbox = sbox
            self.episode_best_operations = self.logic_json["operations"].copy()
            self.episode_best_ds_freq = evaluate_dspectrum_metrics(dspectrum(sbox))[1]
            self.non_improvement_count = 0
        else:
            self.non_improvement_count += 1
        if (self.global_best_metrics is None) or (composite_metrics > self.global_best_metrics):
            self.global_best_metrics = composite_metrics
            self.global_best_score = composite_metrics[0]
            self.global_best_sbox = sbox
            self.global_best_operations = self.logic_json["operations"].copy()
            self.global_best_ds_freq = evaluate_dspectrum_metrics(dspectrum(sbox))[1]

        # --- End Composite Metric Update Block ---

        # Compute reward using the chosen reward function.
        if self.reward_mode == "spectral_combined":
            reward = self.reward_spectral_combined(nonl, self.prev_nonlinearity, du, self.prev_du, ds_val, ls_val, self.non_improvement_count)
        elif self.reward_mode == "spectral_du":
            reward = self.reward_spectral_du(nonl, self.prev_nonlinearity, du, self.prev_du, ds_val, ls_val, self.non_improvement_count)
        elif self.reward_mode == "spectral_nonl":
            reward = self.reward_spectral_nonl(nonl, self.prev_nonlinearity, du, self.prev_du, ds_val, ls_val, self.non_improvement_count)
        elif self.reward_mode == "complexity":
            reward = self.reward_complexity(nonl, self.prev_nonlinearity, du, self.prev_du)
        elif self.reward_mode == "spectral_linear":
            reward = self.reward_spectral_linear(nonl, self.prev_nonlinearity, du, self.prev_du, ds_val, ls_val, self.non_improvement_count)
        elif self.reward_mode == "spectral_linear_du":
            reward = self.reward_spectral_linear_du(nonl, self.prev_nonlinearity, du, self.prev_du, ds_val, ls_val, self.non_improvement_count)
        else:
            reward = self.reward_spectral_combined(nonl, self.prev_nonlinearity, du, self.prev_du, ds_val, ls_val, self.non_improvement_count)
        reward -= new_gate_penalty

        if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            self.prev_linearity = lin
        else:
            self.prev_nonlinearity = nonl
        self.prev_du = du

        progress_entry = {'gate_count': current_gate_count, 'du': du}
        if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            progress_entry["linearity"] = metric
        else:
            progress_entry["nonlinearity"] = metric
        self.progress_history.append(progress_entry)

        terminated = self.non_improvement_count >= self.non_improvement_limit
        truncated = False

        current_ds_freq = evaluate_dspectrum_metrics(dspectrum(sbox))[1]
        print(f"\n=== Step {self.current_step} ===")
        print("S-box (Hex):")
        print(log_sbox_hex(sbox))
        print("Gate Count:", current_gate_count)
        print("\nSequence of Applied Operations:")
        for op in self.logic_json["operations"]:
            print(op)
        if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            print(f"\nMetrics -> DU: {du}, Linearity: {lin}")
        else:
            print(f"\nMetrics -> DU: {du}, Nonlinearity: {nonl}")
        print(f"Reward: {reward} (Mode: {self.reward_mode})")
        print(f"Current Score: {score} (DSpectrum worst freq: {current_ds_freq}) | Episode Best: {self.episode_best_score} | Global Best: {self.global_best_score}")
        print(f"Non-improvement Count: {self.non_improvement_count} (Terminate if >= {self.non_improvement_limit})")

        if terminated:
            du_best, ds_val_best, ls_val_best, nonl_best, lin_best = get_sb_props(self.episode_best_sbox)
            worst_diff_best, worst_freq_best, _ = evaluate_dspectrum_metrics(ds_val_best)
            worst_walsh_best, worst_walsh_freq_best, _ = evaluate_lspectrum_metrics(ls_val_best)
            def count_ops(ops, gate):
                return sum(1 for op in ops if op['type'] == gate)
            episode_entry = {
                'gate_count': len(self.episode_best_operations),
                'du': du_best,
                'worst_walsh_freq': worst_walsh_freq_best,
                'worst_dspec_freq': worst_freq_best,
                'XOR': count_ops(self.episode_best_operations, "XOR"),
                'TOFFOLI': count_ops(self.episode_best_operations, "TOFFOLI"),
                'NOT': count_ops(self.episode_best_operations, "NOT"),
                'FREDKIN': count_ops(self.episode_best_operations, "FREDKIN"),
            }
            if self.reward_mode in ["spectral_linear", "spectral_linear_du"]:
                episode_entry["linearity"] = lin_best
            else:
                episode_entry["nonlinearity"] = nonl_best
            self.episode_history.append(episode_entry)
            self.all_progress_histories.append(self.progress_history)

        return np.array(self.operation_counts, dtype=np.int32), reward, terminated, truncated, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Operation Counts: {self.operation_counts}")
        print("Applied Operations:")
        for op in self.logic_json["operations"]:
            print(op)
        print("Total Gates Used:", len(self.logic_json["operations"]))

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
