import os
from stable_baselines3 import PPO
import torch.nn as nn
from sboxEnv import SBoxEnv
from sbox_utils import get_sb_props, log_sbox_hex
from callbacks import StatsCallback, final_stats, final_stats_progress

custom_reward_config = {
    'w_nonl': 2.0,
    'w_du': 1.5,
    'w_gate': 0.15,
    'w_explore': 0.8,
    'w_no_improve': 0.005,
    'nonl_bonus_threshold': 98,
    'nonl_bonus_factor': 120,
    'du_bonus_threshold': 10,
    'du_bonus_factor': 110,
    'nonl_bonus_threshold2': 100,
    'nonl_bonus_factor2': 1000,
    'du_bonus_threshold2': 8,
    'du_bonus_factor2': 1000,
    'w_toffoli': 2,
    'w_fredkin': 2,
    'spectral_du_weight': 0.005,
    'spectral_nonl_weight': 0.005,
    "spectral_linear_weight": 0.005,
}

hyperparams = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 32,
    "clip_range": 0.2,
    "ent_coef": 0.02,
    "gamma": 0.99,
    "n_epochs": 5,
}
policy_kwargs = dict(
    activation_fn=nn.LeakyReLU,
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
)
total_timesteps = 1000
checkpoint_interval = 500

env_spectral = SBoxEnv(max_steps=100000, reward_mode="spectral_combined",
                       init_random_ops=True, random_ops_count=0,
                       non_improvement_limit=50, reward_config=custom_reward_config,
                       allowed_gates=["XOR", "FREDKIN", "TOFFOLI"])

if os.path.exists("aaNovenaSpecUpdatingSiLU1_spectralDUMC_checkpoint_760000.zip"):
    print("\nLoading existing Spectral Agent...")
    model_spectral = PPO.load("aaNovenaSpecUpdatingSiLU1_spectralDUMC_checkpoint_760000.zip", env=env_spectral)
else:
    print("\nCreating new Spectral Agent...")
    model_spectral = PPO("MlpPolicy", env_spectral, verbose=1, policy_kwargs=policy_kwargs, device="cuda",
                         **hyperparams)

stats_callback_spectral = StatsCallback("spectralDUMC", checkpoint_interval, verbose=1)
model_spectral.learn(total_timesteps=total_timesteps, callback=stats_callback_spectral)
model_spectral.save(env_spectral.model_save_path)

best_sbox_spectral = env_spectral.global_best_sbox
du_spec, ds_spec, ls_spec, nonl_spec, lin_spec = get_sb_props(best_sbox_spectral)
if env_spectral.reward_mode in ["spectral_linear", "spectral_linear_du"]:
    overall_score_spec = lin_spec - du_spec
    metric_str = f"Linearity: {lin_spec}"
else:
    overall_score_spec = nonl_spec - du_spec
    metric_str = f"Nonlinearity: {nonl_spec}"
gate_count_spec = len(env_spectral.global_best_operations)
results_filename_spectral = f"aaNovenaSpec1111UpdatingSiLU_{env_spectral.file_prefix}_sbox_results_spectral_TEST.txt"
with open(results_filename_spectral, "w") as f:
    f.write("\n--- Spectral Agent Best S-box ---\n")
    f.write(log_sbox_hex(best_sbox_spectral) + "\n")
    f.write(f"{metric_str}, Differential Uniformity: {du_spec}, Overall Score: {overall_score_spec}\n")
    f.write("Total Gates Used: " + str(gate_count_spec) + "\n")
    f.write("Sequence of Operations:\n")
    for op in env_spectral.global_best_operations:
        f.write(str(op) + "\n")
print(f"\nSpectral Agent results have been saved to {results_filename_spectral}")

final_stats(env_spectral, "spectralDUMC")
final_stats_progress(env_spectral, "spectralDUMC")

print("\nFinal CSV, JSON, and graphs have been generated for the Spectral Agent.")
