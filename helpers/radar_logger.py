from io import BytesIO
from typing import Dict, List, Optional
import copy

# Use non-interactive backend to avoid Tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pymoo.indicators.hv import HV
from torch.utils.tensorboard import SummaryWriter

from helpers.utils import configure_seaborn


class RadarLogger:
    """
    TensorBoard logger for Radar Neural Architecture Search.
    
    Logs:
    - Pareto front evolution (global and current, colored by policy)
    - Hypervolume metric
    - Policy probability distributions
    - Architecture search statistics
    """

    def __init__(self, config):
        self.config = config
        self.log_dir = config.log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Problem settings
        self.n_objectives = config.problem.n_objectives
        self.nadir = [1.0, 0.5][:self.n_objectives]  # Max loss=1, Max latency=0.5s
        
        # Logging intervals
        self.plot_interval = getattr(config.search, 'plot_interval', 1)
        
        # State tracking
        self.global_optimal_set = None
        self.step = 0
        
        # Configure matplotlib style
        configure_seaborn(grid=False)

    def log_step(self, loggers: dict):
        """
        Main logging entry point.
        
        Expected loggers dict:
        {
            'step': int,
            'optimal_set': Population,  # Current Pareto set
            'policy_manager': PolicyManager,
            'level': int,
            'iteration': int,
        }
        """
        step = loggers.get("step")
        if step is None:
            return
        
        self.step = step
        optimal_set = loggers.get("optimal_set")
        global_pareto_front = loggers.get("global_pareto_front")
        policy_manager = loggers.get("policy_manager")
        level = loggers.get("level", 0)
        iteration = loggers.get("iteration", 0)
        
        # 1. Log Scalar Metrics
        self._log_scalars(step, optimal_set, policy_manager, level, iteration)
        
        # 2. Log Visualizations (less frequently)
        if step % self.plot_interval == 0:
            self._log_visualizations(step, optimal_set, policy_manager, global_pareto_front)

    # =========================================================================
    #  SCALAR LOGGING
    # =========================================================================

    def _log_scalars(self, step, optimal_set, policy_manager, level, iteration):
        """Log scalar metrics to TensorBoard."""
        
        # Hypervolume
        if optimal_set is not None and len(optimal_set) > 0:
            self._log_hypervolume(step, optimal_set, prefix="current")
        
        if self.global_optimal_set is not None and len(self.global_optimal_set) > 0:
            self._log_hypervolume(step, self.global_optimal_set, prefix="global")
        
        # Pareto set size
        if optimal_set is not None:
            self.writer.add_scalar("search/pareto_set_size", len(optimal_set), step)
        
        if self.global_optimal_set is not None:
            self.writer.add_scalar("search/global_pareto_set_size", len(self.global_optimal_set), step)
        
        # Best objectives
        if optimal_set is not None and len(optimal_set) > 0:
            obj_values = np.array([ind.get("F") for ind in optimal_set])
            for i in range(self.n_objectives):
                obj_name = "loss" if i == 0 else "latency"
                self.writer.add_scalar(f"objectives/best_{obj_name}", obj_values[:, i].min(), step)
        
        # Policy weights distribution
        if policy_manager is not None:
            for pol_idx, weight in policy_manager.weights.items():
                self.writer.add_scalar(f"policies/weight_policy_{pol_idx}", weight, step)
            
            # Policy sizes (number of entries)
            for pol_idx, policy in policy_manager.policies.items():
                self.writer.add_scalar(f"policies/size_policy_{pol_idx}", len(policy), step)

    def _log_hypervolume(self, step, population, prefix=""):
        """Compute and log hypervolume indicator."""
        try:
            obj_values = np.array([ind.get("F") for ind in population])
            ref_point = np.array(self.nadir)
            
            # Filter points that are dominated by reference
            valid_mask = np.all(obj_values < ref_point, axis=1)
            if valid_mask.sum() == 0:
                return
                
            hv_indicator = HV(ref_point=ref_point)
            hv = hv_indicator(obj_values[valid_mask])
            
            tag = f"metrics/{prefix}_hypervolume" if prefix else "metrics/hypervolume"
            self.writer.add_scalar(tag, hv, step)
        except Exception as e:
            print(f"HV computation error: {e}")

    # =========================================================================
    #  VISUALIZATIONS
    # =========================================================================

    def _log_visualizations(self, step, optimal_set, policy_manager, global_optimal_set):
        """Log all visual plots."""
        
        # Pareto Front
        if optimal_set is not None and len(optimal_set) > 0:
            self._log_pareto_front(step, optimal_set, global_optimal_set)
        
        # Policy Probabilities
        if policy_manager is not None:
            self._log_policy_heatmap(step, policy_manager)

    def _log_pareto_front(self, step, optimal_set, global_optimal_set):
        """
        Plot Pareto front with points colored by policy.
        Shows both current batch and global archive.
        """
        if self.n_objectives != 2:
            return  # Only support 2D plots for now
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Define colors for policies
            policy_colors = plt.cm.Set1(np.linspace(0, 1, 10))
            
            # Plot global archive (background, smaller, semi-transparent)
            if global_optimal_set is not None and len(global_optimal_set) > 0:
                f_global = np.array([ind.get("F") for ind in global_optimal_set])
                p_global = np.array([ind.get("P") for ind in global_optimal_set])
                
                for pol_idx in np.unique(p_global):
                    mask = p_global == pol_idx
                    ax.scatter(
                        f_global[mask, 0], f_global[mask, 1],
                        c=[policy_colors[int(pol_idx) % 10]],
                        s=60, alpha=0.3, marker='o',
                        label=f'Global P{int(pol_idx)}' if mask.sum() > 0 else None
                    )
            
            # Plot current batch (foreground, larger, solid)
            f_current = np.array([ind.get("F") for ind in optimal_set])
            p_current = np.array([ind.get("P") for ind in optimal_set])
            
            for pol_idx in np.unique(p_current):
                mask = p_current == pol_idx
                ax.scatter(
                    f_current[mask, 0], f_current[mask, 1],
                    c=[policy_colors[int(pol_idx) % 10]],
                    s=120, alpha=0.9, marker='*', edgecolors='black', linewidths=0.5,
                    label=f'Current P{int(pol_idx)}'
                )
            
            # # Reference point (nadir)
            # ax.axvline(x=self.nadir[0], color='red', linestyle='--', alpha=0.5, label='Nadir')
            # ax.axhline(y=self.nadir[1], color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel("Loss (Dice)", fontsize=12)
            ax.set_ylabel("Latency (s)", fontsize=12)
            ax.set_title(f"Pareto Front Evolution (Step {step})", fontsize=14)
            ax.legend(loc='upper right', fontsize=9)
            # ax.set_xlim(0, self.nadir[0] * 1.1)
            # ax.set_ylim(0, self.nadir[1] * 1.1)
            
            self._save_figure_to_tensorboard(fig, "visuals/pareto_front", step)
            
        except Exception as e:
            print(f"Pareto front plotting error: {e}")

    def _log_policy_heatmap(self, step, policy_manager):
        """
        Visualize policy probabilities as a heatmap.
        Shows action preferences for each policy.
        """
        try:
            n_policies = len(policy_manager.policies)
            if n_policies == 0:
                return
            
            # Collect all unique action codes across all policies
            all_codes = set()
            for policy in policy_manager.policies.values():
                all_codes.update(policy.keys())
            
            if len(all_codes) == 0:
                return
            
            # Sort codes for consistent ordering
            all_codes = sorted(all_codes, key=lambda x: str(x))
            n_actions = len(all_codes)
            
            # Limit to most important actions if too many
            max_actions = 50
            if n_actions > max_actions:
                # Select actions with highest variance across policies
                action_values = np.zeros((n_policies, n_actions))
                for i, policy in enumerate(policy_manager.policies.values()):
                    for j, code in enumerate(all_codes):
                        action_values[i, j] = policy.get(code, 0)
                variances = action_values.var(axis=0)
                top_indices = np.argsort(variances)[-max_actions:]
                all_codes = [all_codes[i] for i in sorted(top_indices)]
                n_actions = len(all_codes)
            
            # Build probability matrix
            prob_matrix = np.zeros((n_policies, n_actions))
            
            for i, (pol_idx, policy) in enumerate(policy_manager.policies.items()):
                for j, code in enumerate(all_codes):
                    prob_matrix[i, j] = policy.get(code, 0)
            
            # Apply softmax per policy for visualization
            exp_matrix = np.exp(prob_matrix - prob_matrix.max(axis=1, keepdims=True))
            prob_matrix_norm = exp_matrix / exp_matrix.sum(axis=1, keepdims=True)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(max(12, n_actions * 0.3), 4 + n_policies * 0.5))
            
            im = ax.imshow(prob_matrix_norm, aspect='auto', cmap='YlOrRd', vmin=0)
            
            # Labels
            ax.set_yticks(range(n_policies))
            ax.set_yticklabels([f"Policy {i}" for i in policy_manager.policies.keys()])
            
            # X-axis: show abbreviated action names
            if n_actions <= 30:
                ax.set_xticks(range(n_actions))
                labels = [self._format_action_code(c) for c in all_codes]
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xlabel("Actions (sorted by policy variance)")
            
            ax.set_title(f"Policy Action Probabilities (Step {step})", fontsize=14)
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Probability", fontsize=10)
            
            plt.tight_layout()
            self._save_figure_to_tensorboard(fig, "visuals/policy_heatmap", step)
            
        except Exception as e:
            print(f"Policy heatmap error: {e}")

    def _format_action_code(self, code):
        """Format action code for display."""
        if isinstance(code, tuple):
            # Extract meaningful parts
            parts = []
            for item in code:
                if isinstance(item, tuple) and len(item) == 2:
                    name, val = item
                    # Abbreviate
                    name = name.replace('encoder_', 'E').replace('decoder_', 'D')
                    name = name.replace('_channels', '').replace('bottleneck', 'B')
                    name = name.replace('edge_', 'e')
                    parts.append(f"{name}:{val}")
            return '|'.join(parts[-2:]) if len(parts) > 2 else '|'.join(parts)
        return str(code)[:15]

    # =========================================================================
    #  UTILITIES
    # =========================================================================

    def _save_figure_to_tensorboard(self, fig, tag, step):
        """Convert matplotlib figure to TensorBoard image."""
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img_np = np.array(img)
        
        # Handle RGBA -> RGB if needed
        if img_np.shape[-1] == 4:
            img_np = img_np[:, :, :3]
        
        # HWC -> CHW for TensorBoard
        img_np = img_np.transpose(2, 0, 1)
        self.writer.add_image(tag, img_np, step)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()