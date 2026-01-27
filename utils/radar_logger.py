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
from utils.helpers import configure_seaborn

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