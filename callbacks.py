import json
import csv
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from sbox_utils import get_sb_props, log_sbox_hex


def final_stats(agent_env, agent_type, timestep=None):
    """
    Compute summary statistics from the episode history and save them to a JSON file.
    The 'timestep' parameter (e.g., current num_timesteps) is appended to the filenames
    so that they differ only in timesteps.
    Returns a dictionary with averages and medians for each metric.
    """
    prefix = agent_env.file_prefix
    # Append the timestep (if provided) as the only difference in filenames.
    suffix = f"_{timestep}" if timestep is not None else ""

    if agent_env.reward_mode in ["spectral_linear", "spectral_linear_du"]:
        metric_name = "linearity"
    else:
        metric_name = "nonlinearity"
    metrics = [metric_name, 'du', 'worst_walsh_freq', 'worst_dspec_freq',
               'gate_count', 'XOR', 'TOFFOLI', 'NOT', 'FREDKIN']
    history_data = {}
    for m in metrics:
        # Only include episodes that have this metric.
        history_data[m] = np.array([float(ep[m]) for ep in agent_env.episode_history if m in ep])
    summary = {}
    for m in metrics:
        summary[m] = {
            'average': float(np.mean(history_data[m])),
            'median': float(np.median(history_data[m]))
        }
    summary_filename = f"zopravaSpecpdatingSiLU_{prefix}_final_episode_summary_{agent_type}{suffix}.txt"
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=4)

    csv_filename = f"zopravaSpecpdatingSiLU_{prefix}_final_episode_history_{agent_type}{suffix}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=agent_env.episode_history[0].keys())
        writer.writeheader()
        for row in agent_env.episode_history:
            writer.writerow(row)
    episodes = np.arange(1, len(agent_env.episode_history) + 1)
    # Generate plot: Metric vs. Episodes.
    """
    for m in metrics:
        plt.figure()
        plt.plot(episodes, history_data[m], marker='o', linestyle='-')
        plt.title(f"{prefix}_{agent_type.upper()} Agent: {m} over Episodes{suffix}")
        plt.xlabel("Episode")
        plt.ylabel(m)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"zopravaSpecpdatingSiLU_{prefix}_{agent_type}_{m}_over_episodes{suffix}.png")
        plt.close()

    # Generate histogram plot.
    for m in metrics:
        plt.figure()
        plt.hist(history_data[m], bins='auto', edgecolor='black')
        plt.title(f"{prefix}_{agent_type.upper()} Agent: {m} Histogram{suffix}")
        plt.xlabel(m)
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"zopravaSpecpdatingSiLU_{prefix}_{agent_type}_{m}_histogram{suffix}.png")
        plt.close()

    # Generate frequency (bar) plot.
    for m in metrics:
        counts = Counter(history_data[m])
        keys = sorted(counts.keys())
        values = [counts[k] for k in keys]
        plt.figure()
        plt.bar(keys, values, edgecolor='black')
        plt.title(f"{prefix}_{agent_type.upper()} Agent: {m} Frequency{suffix}")
        plt.xlabel(m)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"zopravaSpecpdatingSiLU_{prefix}_{agent_type}_{m}_frequency{suffix}.png")
        plt.close()

    print(f"Final summary statistics and graphs generated for {agent_type} with prefix {prefix}{suffix}.")
    """
    return summary


def final_stats_progress(agent_env, agent_type, timestep=None):
    """
    Generate and save progression graphs comparing the first and last episodes.
    The 'timestep' parameter is appended to the filenames so that the names differ only by timesteps.
    """
    prefix = agent_env.file_prefix
    suffix = f"_{timestep}" if timestep is not None else ""

    if not agent_env.all_progress_histories:
        print(f"No progress histories recorded for {agent_type}.")
        return
    first = agent_env.all_progress_histories[0]
    last = agent_env.all_progress_histories[-1]

    def extract_progress(progress_list):
        gate = np.array([float(entry['gate_count']) for entry in progress_list])
        if agent_env.reward_mode in ["spectral_linear", "spectral_linear_du"]:
            metric = np.array([float(entry.get("linearity", 0)) for entry in progress_list])
        else:
            metric = np.array([float(entry.get("nonlinearity", 0)) for entry in progress_list])
        du = np.array([float(entry['du']) for entry in progress_list])
        return gate, metric, du

    gate_first, metric_first, du_first = extract_progress(first)
    gate_last, metric_last, du_last = extract_progress(last)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(gate_first, metric_first, marker='o', linestyle='-', label="First Episode")
    plt.plot(gate_last, metric_last, marker='x', linestyle='--', label="Last Episode")
    plt.xlabel("Number of Gates")
    ylabel = "Linearity" if agent_env.reward_mode in ["spectral_linear", "spectral_linear_du"] else "Nonlinearity"
    plt.ylabel(ylabel)
    plt.title(f"{prefix}_{agent_env.reward_mode.upper()} Agent: Progression{suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"zopravaSpecpdatingSiLU_{prefix}_{agent_env.reward_mode}_progress{suffix}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(gate_first, du_first, marker='o', linestyle='-', label="First Episode")
    plt.plot(gate_last, du_last, marker='x', linestyle='--', label="Last Episode")
    plt.xlabel("Number of Gates")
    plt.ylabel("Differential Uniformity")
    plt.title(f"{prefix}_{agent_env.reward_mode.upper()} DU Progression{suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"zopravaSpecpdatingSiLU_{prefix}_{agent_env.reward_mode}_du_progress{suffix}.png")
    plt.close()

    print(f"Progress graphs generated for {agent_env.reward_mode} with prefix {prefix}{suffix}.")
    """

class StatsCallback(BaseCallback):
    """
    A custom callback that, at specified intervals, saves a checkpoint,
    computes and prints summary statistics (with filenames differing only by timestep),
    and saves the global best S-box.
    """

    def __init__(self, agent_type, checkpoint_interval, verbose=0):
        super(StatsCallback, self).__init__(verbose)
        self.agent_type = agent_type
        self.checkpoint_interval = checkpoint_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_interval == 0:
            timestep = self.num_timesteps  # This number will differentiate the filenames.
            checkpoint_path = f"zopravaSpecpdatingSiLU1_{self.agent_type}_checkpoint_{timestep}.zip"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"[Checkpoint] Step {timestep}: Model saved as {checkpoint_path}")
            env = self.training_env.envs[0].unwrapped

            # Compute final stats and progress graphs using the current timestep for unique naming.
            summary_stats = final_stats(env, self.agent_type, timestep=timestep)
            final_stats_progress(env, self.agent_type, timestep=timestep)

            if self.verbose > 0:
                print("[Checkpoint] Summary Statistics:")
                print(json.dumps(summary_stats, indent=4))

            # Save global best S-box information.
            if env.global_best_sbox is not None:
                du_best, ds_val_best, ls_val_best, nonl_best, lin_best = get_sb_props(env.global_best_sbox)
                overall_score = (lin_best if env.reward_mode in ["spectral_linear",
                                                                 "spectral_linear_du"] else nonl_best) - du_best
                gate_count = len(env.global_best_operations)
                best_filename = f"zopravaSpecpdatingSiLU1_{env.file_prefix}_global_best_sbox_{timestep}.txt"
                with open(best_filename, "w") as f:
                    f.write("\n--- Global Best S-box ---\n")
                    f.write(log_sbox_hex(env.global_best_sbox) + "\n")
                    if env.reward_mode in ["spectral_linear", "spectral_linear_du"]:
                        f.write(
                            f"Linearity: {lin_best}, Differential Uniformity: {du_best}, Overall Score: {overall_score}\n")
                    else:
                        f.write(
                            f"Nonlinearity: {nonl_best}, Differential Uniformity: {du_best}, Overall Score: {overall_score}\n")
                    f.write("Gate Count: " + str(gate_count) + "\n")
                    f.write("Operations:\n")
                    for op in env.global_best_operations:
                        f.write(str(op) + "\n")
                if self.verbose > 0:
                    print(f"[Checkpoint] Global best S-box saved to {best_filename}")
        return True

    def _on_training_end(self) -> None:
        env = self.training_env.envs[0].unwrapped
        final_stats(env, self.agent_type)
        final_stats_progress(env, self.agent_type)
        if self.verbose > 0:
            print("[Training End] Final stats and progress graphs generated.")