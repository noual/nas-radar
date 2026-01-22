from typing import Dict, List, Optional, Any

import numpy as np
import random


class RadarNode:
    """
    A lightweight representation of a neural architecture configuration.
    
    This class stores architecture choices (channels and cell operations) as 
    dictionaries without instantiating any PyTorch modules. It is used by the
    SuperNet to define which path to use during forward passes.
    
    The cell structure is a DAG with num_nodes nodes. Node 0 is the input node,
    and each subsequent node i receives inputs from all previous nodes j < i
    via edges. Each edge selects one operation from the available candidates.
    
    Attributes:
        in_channels: Number of input channels.
        initial_channels: Number of channels after stem.
        channel_options: List of possible channel values.
        num_encoder_stages: Number of encoder (and decoder) stages.
        num_nodes: Number of nodes in each cell DAG.
        _encoder_channels: List of channel values for each encoder stage.
        _bottleneck_channels: Channel value for the bottleneck.
        _decoder_channels: List of channel values for each decoder stage.
        _cell_operations: Dict mapping edge names to operation keys.
    """
    
    # Available operations for each edge
    OPERATION_KEYS = [
        'conv_1x1',
        'conv_1x3_3x1',
        'depthwise_conv_3x3',
        'conv_1x5_5x1',
        'dilated_conv_3x3_r2',
        'dilated_conv_3x3_r4',
        'identity',
        'none',
    ]

    def __init__(self, problem):
        """
        Initialise a RadarNode with empty architecture choices.
        
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale).
            initial_channels: Number of channels after the stem.
            channel_options: List of possible channel values for each stage.
            num_encoder_stages: Number of encoder (and decoder) stages.
            num_nodes: Number of nodes in each cell DAG.
        """
        self.in_channels = problem.supernet.in_channels
        self.initial_channels = problem.supernet.initial_channels
        self.channel_options = sorted(problem.supernet.channel_options)
        self.num_encoder_stages = problem.supernet.num_encoder_stages
        self.num_decoder_stages = problem.supernet.num_encoder_stages
        self.num_nodes = problem.supernet.num_nodes
        
        # Network-level channel choices (None = not yet chosen)
        self._encoder_channels: List[Optional[int]] = [None] * self.num_encoder_stages
        self._bottleneck_channels: Optional[int] = None
        self._decoder_channels: List[Optional[int]] = [None] * self.num_encoder_stages
        
        # Cell-level operation choices (None = not yet chosen)
        # All cells share the same operations
        # Edge names follow the pattern "edge_{source}_{target}"
        self._cell_operations: Dict[str, Optional[str]] = {}
        self._init_cell_edges()
        self.path = []
    
    def _init_cell_edges(self) -> None:
        """Initialise all cell edges with None (no operation chosen yet)."""
        for target_node in range(1, self.num_nodes):
            for source_node in range(target_node):
                edge_name = f"edge_{source_node}_{target_node}"
                self._cell_operations[edge_name] = None
    
    def _get_edge_names(self) -> List[str]:
        """Return a list of all edge names in topological order."""
        edge_names = []
        for target_node in range(1, self.num_nodes):
            for source_node in range(target_node):
                edge_names.append(f"edge_{source_node}_{target_node}")
        return edge_names

    @property
    def state(self):
        return self.path
    
    @property
    def is_network_terminal(self) -> bool:
        """Check if all network-level channel choices are fixed."""
        if any(ch is None for ch in self._encoder_channels):
            return False
        if self._bottleneck_channels is None:
            return False
        if any(ch is None for ch in self._decoder_channels):
            return False
        return True
    
    @property
    def is_cell_terminal(self) -> bool:
        """Check if all cell-level operation choices are fixed."""
        return all(op is not None for op in self._cell_operations.values())
    
    @property
    def is_terminal(self) -> bool:
        """Check if the architecture is fully specified."""
        return self.is_network_terminal and self.is_cell_terminal
    
    def get_actions_tuples(self) -> List[tuple]:
        """
        Return available actions in sequential order.
        
        Actions are returned in order:
        1. Encoder channel choices (stage 0, 1, 2, ...)
        2. Bottleneck channel choice
        3. Decoder channel choices (stage n-1, n-2, ..., 0)
        4. Cell operation choices (edge_0_1, edge_0_2, edge_1_2, ...)
        
        Returns:
            List of (action_name, value) tuples for the next unfixed choice.
        """
        # First, fix network-level choices
        if not self.is_network_terminal:
            return self._get_network_actions()
        
        # Then, fix cell-level choices
        return self._get_cell_actions()
    
    def _get_network_actions(self) -> List[tuple]:
        """Return available network-level (channel) actions."""
        # Encoder stages in order
        for i in range(self.num_encoder_stages):
            if self._encoder_channels[i] is None:
                return [(f'encoder_{i}_channels', ch) for ch in self.channel_options]
        
        # Bottleneck
        if self._bottleneck_channels is None:
            return [('bottleneck_channels', ch) for ch in self.channel_options]
        
        # Decoder stages in reverse order (matching network architecture)
        for i in range(self.num_decoder_stages - 1, -1, -1):
            if self._decoder_channels[i] is None:
                return [(f'decoder_{i}_channels', ch) for ch in self.channel_options]
        
        return []
    
    def _get_cell_actions(self) -> List[tuple]:
        """Return available cell-level (operation) actions."""
        # Process edges in topological order
        for edge_name in self._get_edge_names():
            if self._cell_operations[edge_name] is None:
                return [(edge_name, op) for op in self.OPERATION_KEYS]
        
        return []
    
    def play_action(self, action: tuple) -> None:
        """
        Execute an action to fix part of the architecture.
        
        Args:
            action: Tuple of (action_name, value).
                   For channels: ('encoder_0_channels', 64)
                   For operations: ('edge_0_1', 'conv_1x1')
        """
        action_name, value = action
        self.path.append(action)
        # Handle encoder channel selection
        if action_name.startswith('encoder_') and action_name.endswith('_channels'):
            stage_idx = int(action_name.split('_')[1])
            if self._encoder_channels[stage_idx] is not None:
                raise ValueError(f"encoder_{stage_idx}_channels already fixed")
            if value not in self.channel_options:
                raise ValueError(f"Invalid channel value: {value}")
            self._encoder_channels[stage_idx] = value
            return
        
        # Handle bottleneck channel selection
        if action_name == 'bottleneck_channels':
            if self._bottleneck_channels is not None:
                raise ValueError("bottleneck_channels already fixed")
            if value not in self.channel_options:
                raise ValueError(f"Invalid channel value: {value}")
            self._bottleneck_channels = value
            return
        
        # Handle decoder channel selection
        if action_name.startswith('decoder_') and action_name.endswith('_channels'):
            stage_idx = int(action_name.split('_')[1])
            if self._decoder_channels[stage_idx] is not None:
                raise ValueError(f"decoder_{stage_idx}_channels already fixed")
            if value not in self.channel_options:
                raise ValueError(f"Invalid channel value: {value}")
            self._decoder_channels[stage_idx] = value
            return
        
        # Handle cell operation selection
        if action_name.startswith('edge_'):
            if action_name not in self._cell_operations:
                raise ValueError(f"Unknown edge: {action_name}")
            if self._cell_operations[action_name] is not None:
                raise ValueError(f"{action_name} already fixed")
            if value not in self.OPERATION_KEYS:
                raise ValueError(f"Invalid operation: {value}")
            self._cell_operations[action_name] = value
            return
        
        raise ValueError(f"Unknown action: {action_name}")
    
    def playout(self, policy: Dict[str, List[float]],
                move_coder: callable, 
                softmax_temp: float = 1.0) -> 'RadarNode':
        while not self.is_terminal:
            available_actions = self.get_actions_tuples()
            policy_values = [policy.get(move_coder(self.state, act), 0) for act in available_actions]
            exp_values = np.exp(np.array(policy_values) / softmax_temp)
            probs = exp_values / np.sum(exp_values)
            action_index = random.choices(np.arange(len(available_actions)), weights=probs)[0]
            self.play_action(available_actions[action_index])
        return self.path            
    
    def get_channel_config(self) -> Dict[str, Any]:
        """
        Return the channel configuration for all stages.
        
        Returns:
            Dictionary with encoder_channels, bottleneck_channels, and decoder_channels.
        """
        return {
            'encoder_channels': list(self._encoder_channels),
            'bottleneck_channels': self._bottleneck_channels,
            'decoder_channels': list(self._decoder_channels)
        }
    
    def get_cell_operations(self) -> Dict[str, str]:
        """
        Return the operations selected for each edge in the cells.
        
        Returns:
            Dictionary mapping edge names to operation keys.
            Returns None if any operation is not yet fixed.
        """
        if not self.is_cell_terminal:
            return None
        return dict(self._cell_operations)
    
    def to_str(self) -> str:
        """
        Generate a concise string representation of the network architecture.
        Format: enc_ch0:enc_ch1:...:btl_ch:dec_ch0:dec_ch1:...|edge_0_1_op:edge_0_2_op:...
        
        Example: 8:16:32:64:32:16:8|c1x1:id:z:c1x3:...
        """
        parts = []
        
        # Part 1: Channel configuration
        channel_parts = []
        
        # Encoder channels
        for ch in self._encoder_channels:
            channel_parts.append(str(ch) if ch is not None else '?')
        
        # Bottleneck channels
        channel_parts.append(str(self._bottleneck_channels) if self._bottleneck_channels is not None else '?')
        
        # Decoder channels
        for ch in self._decoder_channels:
            channel_parts.append(str(ch) if ch is not None else '?')
        
        parts.append(':'.join(channel_parts))
        
        # Part 2: Cell operations
        # Map operation keys to short names
        op_map = {
            'conv_1x1': 'c1x1',
            'conv_1x3_3x1': 'c1x3',
            'depthwise_conv_3x3': 'dw3x3',
            'conv_1x5_5x1': 'c1x5',
            'dilated_conv_3x3_r2': 'd3r2',
            'dilated_conv_3x3_r4': 'd3r4',
            'identity': 'id',
            'none': 'z',
        }
        
        op_parts = []
        for edge_name in self._get_edge_names():
            op = self._cell_operations[edge_name]
            if op is not None:
                op_parts.append(op_map.get(op, op))
            else:
                op_parts.append('?')
        
        parts.append(':'.join(op_parts))
        
        return '|'.join(parts)
    
    def copy(self) -> 'RadarNode':
        """Create a deep copy of this RadarNode."""
        new_node = RadarNode(
            in_channels=self.in_channels,
            initial_channels=self.initial_channels,
            channel_options=self.channel_options,
            num_encoder_stages=self.num_encoder_stages,
            num_nodes=self.num_nodes
        )
        new_node._encoder_channels = list(self._encoder_channels)
        new_node._bottleneck_channels = self._bottleneck_channels
        new_node._decoder_channels = list(self._decoder_channels)
        new_node._cell_operations = dict(self._cell_operations)
        return new_node
    
if __name__ == "__main__":
    # Example usage
    node = RadarNode(in_channels=1, initial_channels=8, channel_options=[8,16,32,64], num_encoder_stages=3, num_nodes=4)
    print("Initial node:", node.to_str())
    
    while not node.is_terminal:
        actions = node.get_actions_tuples()
        print("Available actions:", actions)
        action = actions[0]  # Just pick the first available action for demonstration
        print("Playing action:", action)
        node.play_action(action)
        print("Node after action:", node.to_str())
    
    print("Final architecture:", node.to_str())