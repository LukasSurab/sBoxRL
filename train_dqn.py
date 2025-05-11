import os
from stable_baselines3 import DQN
import torch.nn as nn
from sboxEnv import SBoxEnv
from sbox_utils import get_sb_props, log_sbox_hex
from callbacks import StatsCallback, final_stats, final_stats_progress

# Custom reward configuration (can be tuned further)
custom_reward_config = {
    'w_nonl': 2.0,
    'w_du': 1.5,
    'w_gate': 0.15,
    'w_explore': 0.8,
    'w_no_improve': 0.005,
    'nonl_bonus_threshold': 98,
    'nonl_bonus_factor': 1000,
    'du_bonus_threshold': 10,
    'du_bonus_factor': 500,
    'nonl_bonus_threshold2': 100,
    'nonl_bonus_factor2': 10000,
    'du_bonus_threshold2': 8,
    'du_bonus_factor2': 10000,
    'w_toffoli': 5,
    'w_fredkin': 5,
    'spectral_du_weight': 0.005,
    'spectral_nonl_weight': 0.005,
    "spectral_linear_weight": 0.005,
}

dqn_hyperparams = {
    "learning_rate": 3e-4,  # Example learning rate adjustment.
    "buffer_size": 100000,
    "learning_starts": 2048,
    "batch_size": 32,
    "tau": 0.05,                      # Increased tau for faster soft updates.
    "gamma": 0.99,
    "train_freq": 1,                  # Updating more frequently.
    "target_update_interval": 1000,    # More frequent target network updates.
    "exploration_fraction": 0.2,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.1,
}
# Network architecture for DQN
policy_kwargs = dict(
    activation_fn=nn.LeakyReLU,
    net_arch=[512, 256, 128]
)

total_timesteps = 1000
checkpoint_interval = 500

env_dqn = SBoxEnv(max_steps=100000, reward_mode="spectral_combined",
                     init_random_ops=True, random_ops_count=0,
                     non_improvement_limit=50, reward_config=custom_reward_config,
                     allowed_gates=["XOR", "TOFFOLI", "FREDKIN"])

if os.path.exists("zDeletaedSpecpdatingSiLU1_dqn_checkpoint_475000.zip"):
    print("\nLoading existing DQN Agent...")
    model_dqn = DQN.load("zDeleteadSpecpdatingSiLU1_dqn_checkpoint_475000.zip", env=env_dqn)
else:
    print("\nCreating new DQN Agent...")
    model_dqn = DQN("MlpPolicy", env_dqn, verbose=1, policy_kwargs=policy_kwargs,device="cpu", **dqn_hyperparams)

stats_callback_dqn = StatsCallback("dqn", checkpoint_interval, verbose=1)
model_dqn.learn(total_timesteps=total_timesteps, callback=stats_callback_dqn)
model_dqn.save(env_dqn.model_save_path)

best_sbox_dqn = env_dqn.global_best_sbox
du_dqn, ds_dqn, ls_dqn, nonl_dqn = get_sb_props(best_sbox_dqn)
overall_score_dqn = nonl_dqn - du_dqn
gate_count_dqn = len(env_dqn.global_best_operations)
results_filename_dqn = f"dSiLU_{env_dqn.file_prefix}_sbox_results_dqn_TEST.txt"
with open(results_filename_dqn, "w") as f:
    f.write("\n--- DQN Agent Best S-box ---\n")
    f.write(log_sbox_hex(best_sbox_dqn) + "\n")
    f.write(f"Nonlinearity: {nonl_dqn}, Differential Uniformity: {du_dqn}, Overall Score: {overall_score_dqn}\n")
    f.write("Total Gates Used: " + str(gate_count_dqn) + "\n")
    f.write("Sequence of Operations:\n")
    for op in env_dqn.global_best_operations:
        f.write(str(op) + "\n")
print(f"\nDQN Agent results have been saved to {results_filename_dqn}")

final_stats(env_dqn, "dqn")
final_stats_progress(env_dqn, "dqn")

print("\nFinal CSV, JSON, and graphs have been generated for the DQN Agent.")
