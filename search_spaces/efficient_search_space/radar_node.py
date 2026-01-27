from typing import List, Optional

from pyparsing import Dict

from efficient_search_space.operations import OPERATIONS

class FrugalRadarNode:
    
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
        
        # Network-level channel choices (None = not yet chosen)
        self._encoder_channels: List[Optional[int]] = [None] * self.num_encoder_stages
        self._bottleneck_channels: Optional[int] = None
        self._decoder_channels: List[Optional[int]] = [None] * self.num_encoder_stages
        
        self._encoder_ops: List[Optional[str]] = [None] * self.num_encoder_stages
        self._bottleneck_op: Optional[str] = None
        self._decoder_ops: List[Optional[str]] = [None] * self.num_encoder_stages

        self.path = []

        self.OP_LIST = list(OPERATIONS.keys())

    @property
    def state(self):
        return self.path
    
    @property
    def is_terminal(self) -> bool:
        """Check if the node is terminal (all choices made)."""
        return self.channels_set and self.operations_set
    
    @property
    def channels_set(self) -> bool:
        """Check if all channel choices have been made."""
        return None not in self._encoder_channels + [self._bottleneck_channels] + self._decoder_channels
    
    @property
    def operations_set(self) -> bool:
        """Check if all operation choices have been made."""
        return None not in self._encoder_ops + [self._bottleneck_op] + self._decoder_ops
    
    def get_actions_tuples(self):
        if not self.channels_set:
            for i in range(self.num_encoder_stages):
                if self._encoder_channels[i] is None:
                    return [(f'encoder_{i}_channels', ch) for ch in self.channel_options]
            if self._bottleneck_channels is None:
                return [('bottleneck_channels', ch) for ch in self.channel_options]
            for i in range(self.num_decoder_stages):
                if self._decoder_channels[i] is None:
                    return [(f'decoder_{i}_channels', ch) for ch in self.channel_options]
        elif not self.operations_set:
            for i in range(self.num_encoder_stages):
                if self._encoder_ops[i] is None:
                    return [(f'encoder_{i}_op', op) for op in self.OP_LIST]
            if self._bottleneck_op is None:
                return [(f'bottleneck_op', op) for op in self.OP_LIST]
            for i in range(self.num_decoder_stages):
                if self._decoder_ops[i] is None:
                    return [(f'decoder_{i}_op', op) for op in self.OP_LIST]
        else:
            raise ValueError("All actions have already been set.")
        
    def to_str(self) -> str:
        op_map = {
            'identity': 'id',
            'conv_1x1': 'c1x1',
            'conv_3x3': 'c3x3',
            'sep_conv_3x3': 'sep3x3',       # New: CPU-optimized depthwise separable
            'double_conv_3x3': 'dbl3x3',    # New: Standard U-Net block
            'double_conv_3x3_d2': 'dbl3x3d2',    # New: Standard U-Net block
            'double_conv_3x3_d4': 'dbl3x3d4',    # New: Standard U-Net block
            'res_double_conv_3x3': 'res2c3', # New: Residual double conv
            'mbconv_3x3_no_se': 'mb3ns',   # New: MBConv without Squeeze-and-Excitation
            'none': 'z'                     # Kept for compatibility if zero-ops are used
        }
        parts = []
        for i in range(self.num_encoder_stages):
            parts.append(f"e{i}_ch{self._encoder_channels[i]}_{op_map.get(self._encoder_ops[i], self._encoder_ops[i])}")
        parts.append(f"b_ch{self._bottleneck_channels}_{op_map.get(self._bottleneck_op, self._bottleneck_op)}")
        for i in range(self.num_decoder_stages):
            parts.append(f"d{i}_ch{self._decoder_channels[i]}_{op_map.get(self._decoder_ops[i], self._decoder_ops[i])}")
        return ":".join(parts)
    
    def from_str(self, architecture_str: str):
        op_map_inv = {
            'id': 'identity',
            'c1x1': 'conv_1x1',
            'c3x3': 'conv_3x3',
            'sep3x3': 'sep_conv_3x3',
            'dbl3x3': 'double_conv_3x3',
            'dbl3x3d2': 'double_conv_3x3_d2',
            'dbl3x3d4': 'double_conv_3x3_d4',
            'res2c3': 'res_double_conv_3x3',
            'mb3ns': 'mbconv_3x3_no_se',
            'z': 'none'
        }
        parts = architecture_str.split(":")
        for part in parts:
            if part.startswith('e'):
                stage, rest = part[1:].split('_ch')
                index = int(stage)
                if rest.endswith("mb3_ns"):  # Temporary fix
                    rest = rest[:-6]+"mb3ns"
                ch_str, op_str = rest.split('_')
                ch = int(ch_str)
                op = op_map_inv.get(op_str, op_str)
                self.play_action(f'encoder_{index}_channels', ch)
                self.play_action(f'encoder_{index}_op', op)
            elif part.startswith('b'):
                _, rest = part.split('_ch')
                if rest.endswith("mb3_ns"):  # Temporary fix
                    rest = rest[:-6]+"mb3ns"
                ch_str, op_str = rest.split('_')
                ch = int(ch_str)
                op = op_map_inv.get(op_str, op_str)
                self.play_action('bottleneck_channels', ch)
                self.play_action('bottleneck_op', op)
            elif part.startswith('d'):
                stage, rest = part[1:].split('_ch')
                index = int(stage)
                if rest.endswith("mb3_ns"):  # Temporary fix
                    rest = rest[:-6]+"mb3ns"
                ch_str, op_str = rest.split('_')
                ch = int(ch_str)
                op = op_map_inv.get(op_str, op_str)
                self.play_action(f'decoder_{index}_channels', ch)
                self.play_action(f'decoder_{index}_op', op)
            else:
                raise ValueError(f"Invalid architecture string part: {part}")
        
    def play_action(self, name: str, value):
        if name.startswith('encoder_') and name.endswith('_channels'):
            index = int(name.split('_')[1])
            self._encoder_channels[index] = value
        elif name == 'bottleneck_channels':
            self._bottleneck_channels = value
        elif name.startswith('decoder_') and name.endswith('_channels'):
            index = int(name.split('_')[1])
            self._decoder_channels[index] = value
        elif name.startswith('encoder_') and name.endswith('_op'):
            index = int(name.split('_')[1])
            self._encoder_ops[index] = value
        elif name == 'bottleneck_op':
            self._bottleneck_op = value
        elif name.startswith('decoder_') and name.endswith('_op'):
            index = int(name.split('_')[1])
            self._decoder_ops[index] = value
        else:
            raise ValueError(f"Unknown action name: {name}")
        
        self.path.append((name, value))
        
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    sys.path.append("..")
    from search_spaces.radar import Radar
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "problem": {
            "n_objectives": 2,
            "dataset_path": "./data/train_bth/mat",
            "batch_size": 8,
            "supernet": {
                "in_channels": 1,
                "initial_channels": 8,
                "channel_options": [8, 16, 32, 64, 128],
                "num_encoder_stages": 3,
                "num_nodes": 4,
                "n_steps": 5
                },

        },
        "device": "cpu",
        "seed": 42
    })
    problem = Radar(config)
    node = FrugalRadarNode(problem)
    while not node.is_terminal:
        actions = node.get_actions_tuples()
        print("Available actions:", actions)
        action = actions[0]
        print("Playing action:", action)
        name, value = action
        node.play_action(name, value)
    print("Final architecture state:", node.state)
