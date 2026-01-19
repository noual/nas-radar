from search_space.cell import RadarCell
from search_space.network_structure import RadarNetwork
from nni.nas.nn.pytorch import LayerChoice

class RadarNode:

    def __init__(self, in_channels, initial_channels, channel_options, 
                 num_encoder_stages=3, num_nodes=4):
        self.in_channels = in_channels
        self.initial_channels = initial_channels
        self.channel_options = channel_options
        self.num_encoder_stages = num_encoder_stages
        self.num_nodes = num_nodes

        self.cell_factory = lambda c_in, c_out, num_nodes: RadarCell(c_in, c_out, num_nodes)
        self.network = RadarNetwork(
            in_channels=self.in_channels,
            initial_channels=self.initial_channels,
            channel_options=self.channel_options,
            cell_factory=self.cell_factory,
            num_nodes=self.num_nodes)
        
    @property
    def is_terminal(self):
        network_terminal = self.network.is_terminal
        if not network_terminal:
            return False
        cell_terminal = all(cell.is_terminal for cell in self.network.encoder_cells)
        return network_terminal and cell_terminal
    
        
    def get_actions_tuples(self):
        # Check if the network is terminal
        if not self.network.is_terminal:
            return self.network.get_actions()
        else:
            return self.network.encoder_cells[0].get_actions()
        
    def play_action(self, action: tuple):
        self.network.play_action(action)
    
    def to_str(self):
        """
        Generate a concise string representation of the network architecture.
        Format: enc_ch0:enc_ch1:...:btl_ch:dec_ch0:dec_ch1:...|edge_0_1_op:edge_0_2_op:...
        
        Example: 8:16:32:64:32:16:8|conv_1x1:identity:none:conv_1x3_3x1:...
        """
        parts = []
        
        # Part 1: Channel configuration
        channel_parts = []
        
        # Encoder channels
        for ch in self.network._encoder_channels:
            channel_parts.append(str(ch) if ch is not None else '?')
        
        # Bottleneck channels
        channel_parts.append(str(self.network._bottleneck_channels) if self.network._bottleneck_channels is not None else '?')
        
        # Decoder channels
        for ch in self.network._decoder_channels:
            channel_parts.append(str(ch) if ch is not None else '?')
        
        parts.append(':'.join(channel_parts))
        
        # Part 2: Cell operations (only if cells are created)
        if self.network.encoder_cells[0] is not None:
            cell = self.network.encoder_cells[0]  # All cells are identical
            op_parts = []
            
            for i in range(1, cell.num_nodes):
                for j in range(i):
                    edge_name = f"edge_{j}_{i}"
                    op = cell.layers[f"node_{i}"][edge_name]
                    
                    # Get operation name
                    if isinstance(op, LayerChoice):
                        op_name = '?'
                    else:
                        # Get the actual operation class name and simplify
                        op_class = op.__class__.__name__
                        # Map class names to short names
                        op_map = {
                            'Conv1x1': 'c1x1',
                            'Conv1x3_3x1': 'c1x3',
                            'Conv1x5_5x1': 'c1x5',
                            'DilatedConv3x3_r2': 'd3r2',
                            'DilatedConv3x3_r4': 'd3r4',
                            'Identity': 'id',
                            'Zero': 'z'
                        }
                        op_name = op_map.get(op_class, op_class.lower())
                    
                    op_parts.append(op_name)
            
            parts.append(':'.join(op_parts))
        else:
            parts.append('?')
        
        return '|'.join(parts)
