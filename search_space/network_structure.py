import torch
import torch.nn as nn
import nni
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, MutableModule
from collections import OrderedDict
from search_space.cell import RadarCell, prune_cell


class ChannelAdapter(nn.Module):
    """Adapter module to adjust channel dimensions using 1x1 convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.conv = None
    
    def forward(self, x):
        if self.conv is not None:
            return self.conv(x)
        return x


class RadarNetwork(ModelSpace):
    """
    U-Net style network with flexible number of encoding and decoding stages.
    Uses nni.choice() for channel selection at each stage.
    
    Encoder and decoder cells are created dynamically based on chosen channel values.
    """
    
    def __init__(self, in_channels, initial_channels, channel_options, cell_factory, 
                 num_encoder_stages=3, num_nodes=4):
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            initial_channels: Initial number of channels after first conv
            channel_options: List of possible channel counts for each stage (e.g., [32, 64, 128])
            cell_factory: Function that creates a RadarCell given (C_in, C_out, num_nodes)
            num_encoder_stages: Number of encoding stages (equal to number of decoding stages)
            num_nodes: Number of nodes in each RadarCell
        """
        super().__init__()
        self.in_channels = in_channels
        self.initial_channels = initial_channels
        self.channel_options = channel_options
        self.num_nodes = num_nodes
        self.cell_factory = cell_factory
        self.num_encoder_stages = num_encoder_stages
        self.num_decoder_stages = num_encoder_stages  # Mirror architecture
        
        # Initial convolution to map input to initial_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # Track fixed channel choices for encoder stages
        self._encoder_channels = [None] * num_encoder_stages
        self._bottleneck_channels = None
        self._decoder_channels = [None] * num_encoder_stages
        
        # Create downsampling layers
        self.downsample_layers = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(num_encoder_stages)
        ])
        
        # Create upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
            for _ in range(num_encoder_stages)
        ])
        
        # Stage cells (will be created progressively as channels are chosen)
        self.encoder_cells = nn.ModuleList([None] * num_encoder_stages)
        self.bottleneck = None
        self.decoder_cells = nn.ModuleList([None] * num_encoder_stages)
        self.final_conv = None
    
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Encoding path with skip connections
        encoder_outputs = []
        for i in range(self.num_encoder_stages):
            if self.encoder_cells[i] is None:
                raise RuntimeError(f"encoder_cells[{i}] not created. Fix channel choices using play_action() first.")
            x = self.encoder_cells[i](x)
            encoder_outputs.append(x)
            x = self.downsample_layers[i](x)
        
        # Bottleneck
        if self.bottleneck is None:
            raise RuntimeError("bottleneck not created. Fix channel choices using play_action() first.")
        x = self.bottleneck(x)
        
        # Decoding path with skip connections (reverse order)
        for i in range(self.num_decoder_stages - 1, -1, -1):
            x = self.upsample_layers[i](x)
            # Concatenate with corresponding encoder output
            x = torch.cat([x, encoder_outputs[i]], dim=1)
            if self.decoder_cells[i] is None:
                raise RuntimeError(f"decoder_cells[{i}] not created. Fix channel choices using play_action() first.")
            x = self.decoder_cells[i](x)
        
        # Final output
        if self.final_conv is None:
            raise RuntimeError("final_conv not created. Fix channel choices using play_action() first.")
        x = self.final_conv(x)
        return x
    
    @property
    def is_terminal(self):
        """
        Check if the architecture is fully determined at the network level.
        Returns True when all channel choices are fixed (not checking cell-level operations).
        """
        # Check if all encoder channel choices are made
        if any(ch is None for ch in self._encoder_channels):
            return False
        
        # Check if bottleneck channel choice is made
        if self._bottleneck_channels is None:
            return False
        
        # Check if all decoder channel choices are made
        if any(ch is None for ch in self._decoder_channels):
            return False
        
        return True
    
    
    def get_actions(self):
        """
        Return available actions in sequential order.
        Only returns channel selection actions (not cell-level operations).
        
        Returns:
            List of tuples representing available actions:
            - Channel selection: ('encoder_0_channels', channel_value)
        """
        actions = []
        
        # Process encoder stages
        for i in range(self.num_encoder_stages):
            # Fix encoder channel choice
            if self._encoder_channels[i] is None:
                for ch in self.channel_options:
                    actions.append((f'encoder_{i}_channels', ch))
                return actions
        
        # Fix bottleneck channel choice
        if self._bottleneck_channels is None:
            for ch in self.channel_options:
                actions.append(('bottleneck_channels', ch))
            return actions
        
        # Process decoder stages (reverse order matches network architecture)
        for i in range(self.num_decoder_stages - 1, -1, -1):
            # Fix decoder channel choice
            if self._decoder_channels[i] is None:
                for ch in self.channel_options:
                    actions.append((f'decoder_{i}_channels', ch))
                return actions
        
        # All stages complete
        return actions
    
    
    def play_action(self, action: tuple):
        """
        Execute an action to fix part of the architecture.
        
        Args:
            action: Tuple of (action_name, value)
                - For channel selection: ('encoder_0_channels', 64)
                - For cell operations: ('encoder_0:node_1:edge_0_1', 'conv_1x1')
        """
        action_name, value = action
        
        # Handle encoder channel selection
        if action_name.startswith('encoder_') and action_name.endswith('_channels'):
            # Extract stage index
            stage_idx = int(action_name.split('_')[1])
            if self._encoder_channels[stage_idx] is not None:
                raise ValueError(f"encoder_{stage_idx}_channels already fixed")
            
            self._encoder_channels[stage_idx] = value
            
            # Determine input channels for this encoder stage
            if stage_idx == 0:
                in_channels = self.initial_channels
            else:
                in_channels = self._encoder_channels[stage_idx - 1]
            
            # Create encoder cell
            self.encoder_cells[stage_idx] = self.cell_factory(in_channels, value, self.num_nodes)
            return
        
        # Handle bottleneck channel selection
        elif action_name == 'bottleneck_channels':
            if self._bottleneck_channels is not None:
                raise ValueError("bottleneck_channels already fixed")
            self._bottleneck_channels = value
            
            # Input channels from last encoder stage
            last_enc_channels = self._encoder_channels[-1]
            self.bottleneck = self.cell_factory(last_enc_channels, value, self.num_nodes)
            return
        
        # Handle decoder channel selection
        elif action_name.startswith('decoder_') and action_name.endswith('_channels'):
            # Extract stage index
            stage_idx = int(action_name.split('_')[1])
            if self._decoder_channels[stage_idx] is not None:
                raise ValueError(f"decoder_{stage_idx}_channels already fixed")
            
            self._decoder_channels[stage_idx] = value
            
            # Determine input channels for decoder (concatenated: upsampled + skip connection)
            if stage_idx == self.num_decoder_stages - 1:
                # First decoder receives bottleneck output + last encoder output
                upsample_channels = self._bottleneck_channels
            else:
                # Other decoders receive previous decoder output + corresponding encoder output
                upsample_channels = self._decoder_channels[stage_idx + 1]
            
            skip_channels = self._encoder_channels[stage_idx]
            in_channels = upsample_channels + skip_channels
            
            # Create decoder cell
            self.decoder_cells[stage_idx] = self.cell_factory(in_channels, value, self.num_nodes)
            
            # If this is the last decoder (index 0), also create final conv
            if stage_idx == 0:
                self.final_conv = nn.Conv2d(value, self.in_channels, kernel_size=1)
            return
        
        # Handle cell-level actions (operations within cells)
        # Apply the action to ALL cells in the network since they are identical
        elif ':' in action_name:
            # Extract the cell action part (after the stage prefix)
            # e.g., 'encoder_0:node_1:edge_0_1' -> 'node_1:edge_0_1'
            
            
            # Apply to all encoder cells
            for cell in self.encoder_cells:
                if cell is not None:
                    cell.play_action((action_name, value))
            
            # Apply to bottleneck
            if self.bottleneck is not None:
                self.bottleneck.play_action((action_name, value))
            
            # Apply to all decoder cells
            for cell in self.decoder_cells:
                if cell is not None:
                    cell.play_action((action_name, value))
            
            return
        
        raise ValueError(f"Unknown action: {action_name}")
    
    
    def prune_cells(self):
        """
        Prune all cells in the network by removing inactive nodes.
        Only prunes cells that are terminal (all operations fixed).
        Replaces each cell with its pruned version.
        """
        # Prune encoder cells
        for i in range(self.num_encoder_stages):
            if self.encoder_cells[i] is not None and self.encoder_cells[i].is_terminal:
                self.encoder_cells[i] = prune_cell(self.encoder_cells[i])
        
        # Prune bottleneck
        if self.bottleneck is not None and self.bottleneck.is_terminal:
            self.bottleneck = prune_cell(self.bottleneck)
        
        # Prune decoder cells
        for i in range(self.num_decoder_stages):
            if self.decoder_cells[i] is not None and self.decoder_cells[i].is_terminal:
                self.decoder_cells[i] = prune_cell(self.decoder_cells[i])
