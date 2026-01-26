"""
SuperNet Module for Radar Neural Architecture Search.

This module implements an over-parameterised network (SuperNet) that acts as a
weight container for all possible subnetworks within the search space. The
SuperNet stores weights for the maximal architecture configuration and supports
efficient subnet sampling with weight slicing (Slimmable Networks approach).

Classes:
    SuperNet: Main class for weight storage and subnet sampling.

Example Usage:
    >>> from search_spaces.supernet import SuperNet
    >>> from search_spaces.radar_node import RadarNode
    >>> 
    >>> # Create a SuperNet
    >>> supernet = SuperNet(
    ...     in_channels=1,
    ...     initial_channels=8,
    ...     channel_options=[8, 16, 32, 64],
    ...     num_encoder_stages=3,
    ...     num_nodes=4
    ... )
    >>> 
    >>> # Sample a subnet from a RadarNode
    >>> node = RadarNode(1, 8, [8, 16, 32, 64], num_encoder_stages=3, num_nodes=4)
    >>> # ... configure node with actions ...
    >>> subnet = supernet.sample(node)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import copy

from search_spaces.dag_search_space.cell import RadarCell
from search_spaces.dag_search_space.network_structure import (
    RadarNetwork, 
    MaximalRadarNetwork, 
    ChannelAdapter,
    create_stem
)
from search_spaces.dag_search_space.operations import (
    Conv1x1, Conv1x3_3x1, Conv1x5_5x1, 
    DilatedConv3x3_r2, DilatedConv3x3_r4, 
    Identity, Zero, DepthwiseConv3x3,
    get_op_candidates
)


class SuperNet(nn.Module):
    """
    Over-parameterised network for weight sharing in Neural Architecture Search.
    
    The SuperNet maintains a "Maximal Architecture" where every stage uses the
    maximum channel count and every edge contains instantiated modules for all
    possible operations. This serves as the "Weight Bank" from which subnetworks
    inherit their weights.
    
    Weight Sharing Strategy:
        - Channel Slicing: For subnetworks with fewer channels than the maximum,
          weights are sliced from the SuperNet tensors along the channel dimensions.
          For a convolution weight of shape (C_out, C_in, K, K), a subnet with
          (C_sub_out, C_sub_in) channels receives weight[:C_sub_out, :C_sub_in, :, :].
        
        - Operation Selection: Each edge in the subnet selects one operation from
          the SuperNet's operation set for that edge.
    
    Attributes:
        in_channels: Number of input channels (e.g., 1 for grayscale radar data).
        initial_channels: Number of channels after the stem convolution.
        channel_options: List of possible channel counts for architecture search.
        max_channels: Maximum channel count (max of channel_options).
        num_encoder_stages: Number of encoder stages in the U-Net structure.
        num_nodes: Number of nodes per RadarCell DAG.
        maximal_net: The internal MaximalRadarNetwork storing all weights.
    """
    
    def __init__(
        self,
        in_channels: int,
        initial_channels: int,
        channel_options: List[int],
        num_encoder_stages: int = 3,
        num_nodes: int = 4,
        device='cpu'
    ):
        """
        Initialise the SuperNet with the maximal architecture.
        
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale).
            initial_channels: Number of channels produced by the stem.
            channel_options: List of possible channel counts for each stage.
            num_encoder_stages: Number of encoding (and decoding) stages.
            num_nodes: Number of nodes in each RadarCell DAG.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.initial_channels = initial_channels
        self.channel_options = sorted(channel_options)
        self.max_channels = max(channel_options)
        self.num_encoder_stages = num_encoder_stages
        self.num_decoder_stages = num_encoder_stages
        self.num_nodes = num_nodes
        self.device = device
        
        # Create the maximal network (Weight Bank)
        self.maximal_net = MaximalRadarNetwork(
            in_channels=in_channels,
            initial_channels=initial_channels,
            max_channels=self.max_channels,
            num_encoder_stages=num_encoder_stages,
            num_nodes=num_nodes,
            device=device
        )
        
        # Cache operation names for convenience
        self._op_names = list(get_op_candidates(1, 1).keys())
    
    def forward(self, x: torch.Tensor, selected_ops: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """
        Forward pass through the SuperNet (maximal network).
        
        Args:
            x: Input tensor of shape (B, in_channels, H, W).
            selected_ops: Optional dict mapping edge names to operation names.
                         If None, all operations are summed (not recommended
                         for training; use for architecture search exploration).
                         
        Returns:
            Output tensor of shape (B, in_channels, H, W).
        """
        return self.maximal_net(x, selected_ops)
    
    def sample(self, node: 'RadarNode') -> RadarNetwork:
        """
        Sample a subnet from the SuperNet based on a RadarNode configuration.
        
        This method creates an independent RadarNetwork instance with weights
        copied (and sliced where necessary) from the SuperNet. The returned
        network is a standalone module ready for forward passes.
        
        Args:
            node: A RadarNode instance defining the architecture configuration.
                 The node must have all channel choices and cell operations fixed
                 (i.e., node.is_terminal must be True).
                 
        Returns:
            A RadarNetwork instance with weights initialised from the SuperNet.
            
        Raises:
            ValueError: If the node architecture is not fully specified.
        """
        if not node.is_terminal:
            raise ValueError(
                "Cannot sample from a non-terminal RadarNode. "
                "All channel choices and cell operations must be fixed."
            )
        
        # Extract configuration from node (now directly from RadarNode)
        channel_config = node.get_channel_config()
        cell_ops = node.get_cell_operations()
        
        # Create a new independent RadarNetwork
        subnet = self._create_subnet(channel_config, cell_ops)
        
        # Copy weights from SuperNet to subnet
        self._copy_weights_to_subnet(subnet, channel_config, cell_ops)
        
        return subnet
    
    def _create_subnet(
        self, 
        channel_config: Dict[str, Any], 
        cell_ops: Dict[str, str]
    ) -> RadarNetwork:
        """
        Create a new RadarNetwork instance with the specified configuration.
        
        Args:
            channel_config: Dictionary containing encoder_channels, 
                           bottleneck_channels, and decoder_channels.
            cell_ops: Dictionary mapping edge names to operation keys.
            
        Returns:
            A new RadarNetwork instance (without weights copied yet).
        """
        encoder_channels = channel_config['encoder_channels']
        bottleneck_channels = channel_config['bottleneck_channels']
        decoder_channels = channel_config['decoder_channels']
        
        # Create cell factory that produces cells with fixed operations
        def cell_factory(c_in, c_out, num_nodes):
            cell = RadarCell(c_in, c_out, num_nodes, store_all_ops=False)
            # Fix all operations based on cell_ops
            for node_idx, edge_idx, edge_name, edge_module in cell.iterate_edges():
                if edge_name in cell_ops:
                    # cell_ops now contains operation keys directly
                    op_key = cell_ops[edge_name]
                    # Get the input channels for this edge
                    edge_in_channels = c_in if edge_idx == 0 else c_out
                    # Create the operation with correct channels
                    new_op = get_op_candidates(edge_in_channels, c_out)[op_key]
                    cell.set_edge_operation(node_idx, edge_idx, new_op)
            return cell
        
        # Create network structure
        subnet = RadarNetwork(
            in_channels=self.in_channels,
            initial_channels=self.initial_channels,
            channel_options=self.channel_options,
            cell_factory=cell_factory,
            num_encoder_stages=self.num_encoder_stages,
            num_nodes=self.num_nodes
        )
        
        # Apply channel choices to instantiate the network
        for i, ch in enumerate(encoder_channels):
            subnet.play_action((f'encoder_{i}_channels', ch))
        
        subnet.play_action(('bottleneck_channels', bottleneck_channels))
        
        for i in range(self.num_decoder_stages - 1, -1, -1):
            subnet.play_action((f'decoder_{i}_channels', decoder_channels[i]))
        
        return subnet
    
    def _class_name_to_op_key(self, class_name: str) -> str:
        """
        Map an operation class name to its dictionary key.
        
        Args:
            class_name: The class name (e.g., 'Conv1x1', 'DilatedConv3x3_r2').
            
        Returns:
            The corresponding operation key (e.g., 'conv_1x1', 'dilated_conv_3x3_r2').
        """
        mapping = {
            'Conv1x1': 'conv_1x1',
            'Conv1x3_3x1': 'conv_1x3_3x1',
            'Conv1x5_5x1': 'conv_1x5_5x1',
            'DilatedConv3x3_r2': 'dilated_conv_3x3_r2',
            'DilatedConv3x3_r4': 'dilated_conv_3x3_r4',
            'DepthwiseConv3x3': 'depthwise_conv_3x3',
            'Identity': 'identity',
            'Zero': 'none',
        }
        if class_name not in mapping:
            raise ValueError(f"Unknown operation class: {class_name}")
        return mapping[class_name]
    
    def _copy_weights_to_subnet(
        self,
        subnet: RadarNetwork,
        channel_config: Dict[str, Any],
        cell_ops: Dict[str, str]
    ) -> None:
        """
        Copy (and slice) weights from the SuperNet to a subnet.
        
        This method handles the weight slicing logic for slimmable networks:
        - Convolution weights: slice along output and input channel dimensions.
        - BatchNorm parameters: slice along the channel dimension.
        - Bias terms: slice along the output channel dimension.
        
        Args:
            subnet: The target RadarNetwork to receive weights.
            channel_config: Channel configuration of the subnet.
            cell_ops: Operation selections for each edge.
        """
        encoder_channels = channel_config['encoder_channels']
        bottleneck_channels = channel_config['bottleneck_channels']
        decoder_channels = channel_config['decoder_channels']
        
        with torch.no_grad():
            # Copy stem weights (initial_channels is fixed, so direct copy)
            self._copy_stem_weights(subnet)
            
            # Copy encoder cell weights
            for i in range(self.num_encoder_stages):
                if i == 0:
                    c_in = self.initial_channels
                else:
                    c_in = encoder_channels[i - 1]
                c_out = encoder_channels[i]
                
                self._copy_cell_weights(
                    src_cell=self.maximal_net.encoder_cells[i],
                    dst_cell=subnet.encoder_cells[i],
                    c_in_sub=c_in,
                    c_out_sub=c_out,
                    c_in_max=self.max_channels,
                    c_out_max=self.max_channels,
                    cell_ops=cell_ops
                )
            
            # Copy bottleneck weights
            c_in = encoder_channels[-1]
            c_out = bottleneck_channels
            self._copy_cell_weights(
                src_cell=self.maximal_net.bottleneck,
                dst_cell=subnet.bottleneck,
                c_in_sub=c_in,
                c_out_sub=c_out,
                c_in_max=self.max_channels,
                c_out_max=self.max_channels,
                cell_ops=cell_ops
            )
            
            # Copy decoder cell weights
            for i in range(self.num_decoder_stages - 1, -1, -1):
                if i == self.num_decoder_stages - 1:
                    upsample_ch = bottleneck_channels
                else:
                    upsample_ch = decoder_channels[i + 1]
                skip_ch = encoder_channels[i]
                c_in = upsample_ch + skip_ch
                c_out = decoder_channels[i]
                
                # For decoder, max input is 2 * max_channels (concat)
                self._copy_cell_weights(
                    src_cell=self.maximal_net.decoder_cells[i],
                    dst_cell=subnet.decoder_cells[i],
                    c_in_sub=c_in,
                    c_out_sub=c_out,
                    c_in_max=2 * self.max_channels,
                    c_out_max=self.max_channels,
                    cell_ops=cell_ops
                )
            
            # Copy final convolution weights
            self._copy_final_conv_weights(subnet, decoder_channels[0])
    
    def _copy_stem_weights(self, subnet: RadarNetwork) -> None:
        """
        Copy stem convolution weights from SuperNet to subnet.
        
        The stem has fixed initial_channels, so weights are copied directly
        without slicing.
        
        Args:
            subnet: Target network to receive weights.
        """
        # Copy stem conv weight
        src_conv = self.maximal_net.stem[0]  # Conv2d
        dst_conv = subnet.stem[0]
        dst_conv.weight.copy_(src_conv.weight)
        
        # Copy stem BatchNorm
        src_bn = self.maximal_net.stem[1]  # BatchNorm2d
        dst_bn = subnet.stem[1]
        dst_bn.weight.copy_(src_bn.weight)
        dst_bn.bias.copy_(src_bn.bias)
        dst_bn.running_mean.copy_(src_bn.running_mean)
        dst_bn.running_var.copy_(src_bn.running_var)
        if src_bn.num_batches_tracked is not None:
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)
    
    def _copy_cell_weights(
        self,
        src_cell: RadarCell,
        dst_cell: RadarCell,
        c_in_sub: int,
        c_out_sub: int,
        c_in_max: int,
        c_out_max: int,
        cell_ops: Dict[str, str]
    ) -> None:
        """
        Copy weights from a source cell (SuperNet) to a destination cell (subnet).
        
        This method handles the per-edge weight copying with channel slicing.
        
        Args:
            src_cell: Source cell from the SuperNet.
            dst_cell: Destination cell in the subnet.
            c_in_sub: Input channels for the subnet cell.
            c_out_sub: Output channels for the subnet cell.
            c_in_max: Input channels for the SuperNet cell.
            c_out_max: Output channels for the SuperNet cell.
            cell_ops: Dictionary mapping edge names to selected operation keys.
        """
        for node_idx, edge_idx, edge_name, dst_edge_module in dst_cell.iterate_edges():
            # Determine channels for this specific edge
            if edge_idx == 0:
                edge_c_in_sub = c_in_sub
                edge_c_in_max = c_in_max
            else:
                edge_c_in_sub = c_out_sub
                edge_c_in_max = c_out_max
            
            edge_c_out_sub = c_out_sub
            edge_c_out_max = c_out_max
            
            # Get the operation key (cell_ops now contains keys directly)
            op_key = cell_ops[edge_name]
            
            # Get source operation from SuperNet
            src_op = src_cell.get_edge_operation(node_idx, edge_idx, op_key)
            
            # Copy weights with slicing
            self._copy_operation_weights(
                src_op=src_op,
                dst_op=dst_edge_module,
                c_in_sub=edge_c_in_sub,
                c_out_sub=edge_c_out_sub,
                c_in_max=edge_c_in_max,
                c_out_max=edge_c_out_max
            )
    
    def _copy_operation_weights(
        self,
        src_op: nn.Module,
        dst_op: nn.Module,
        c_in_sub: int,
        c_out_sub: int,
        c_in_max: int,
        c_out_max: int
    ) -> None:
        """
        Copy weights from a source operation to a destination operation with slicing.
        
        This method handles the weight slicing logic for different operation types:
        - Conv2d: weight[:c_out_sub, :c_in_sub, :, :]
        - BatchNorm2d: weight[:c_sub], bias[:c_sub], running_mean[:c_sub], running_var[:c_sub]
        - Identity with channel matching: handled via 1x1 conv
        - Zero: no weights to copy
        
        Args:
            src_op: Source operation module from SuperNet.
            dst_op: Destination operation module in subnet.
            c_in_sub: Input channels for subnet operation.
            c_out_sub: Output channels for subnet operation.
            c_in_max: Input channels for SuperNet operation.
            c_out_max: Output channels for SuperNet operation.
        """
        src_type = type(src_op).__name__
        dst_type = type(dst_op).__name__
        
        # Skip Zero operations (no weights)
        if src_type == 'Zero' or dst_type == 'Zero':
            return
        
        # Handle each operation type
        if src_type == 'Conv1x1':
            self._copy_conv_bn_relu(
                src_op.conv, src_op.bn, dst_op.conv, dst_op.bn,
                c_in_sub, c_out_sub
            )
        
        elif src_type == 'Conv1x3_3x1':
            # First conv: 1x3
            self._copy_conv_bn(
                src_op.conv1, src_op.bn1, dst_op.conv1, dst_op.bn1,
                c_in_sub, c_out_sub
            )
            # Second conv: 3x1
            self._copy_conv_bn(
                src_op.conv2, src_op.bn2, dst_op.conv2, dst_op.bn2,
                c_out_sub, c_out_sub  # Internal channels = out_channels
            )
        
        elif src_type == 'Conv1x5_5x1':
            # First conv: 1x5
            self._copy_conv_bn(
                src_op.conv1, src_op.bn1, dst_op.conv1, dst_op.bn1,
                c_in_sub, c_out_sub
            )
            # Second conv: 5x1
            self._copy_conv_bn(
                src_op.conv2, src_op.bn2, dst_op.conv2, dst_op.bn2,
                c_out_sub, c_out_sub
            )
        
        elif src_type in ('DilatedConv3x3_r2', 'DilatedConv3x3_r4'):
            self._copy_conv_bn_relu(
                src_op.conv, src_op.bn, dst_op.conv, dst_op.bn,
                c_in_sub, c_out_sub
            )
        
        elif src_type == 'DepthwiseConv3x3':
            # Depthwise conv: groups = in_channels
            self._copy_depthwise_conv(
                src_op.depthwise, dst_op.depthwise,
                c_in_sub, c_in_max
            )
            # Pointwise conv: 1x1
            self._copy_conv_bn(
                src_op.pointwise, src_op.bn, dst_op.pointwise, dst_op.bn,
                c_in_sub, c_out_sub
            )
        
        elif src_type == 'Identity':
            # Identity may have a 1x1 conv for channel matching
            if src_op.match is not None and dst_op.match is not None:
                self._copy_conv(
                    src_op.match, dst_op.match,
                    c_in_sub, c_out_sub
                )
        
        else:
            raise ValueError(f"Unknown operation type: {src_type}")
    
    def _copy_conv(
        self,
        src_conv: nn.Conv2d,
        dst_conv: nn.Conv2d,
        c_in_sub: int,
        c_out_sub: int
    ) -> None:
        """
        Copy convolution weights with channel slicing.
        
        Args:
            src_conv: Source Conv2d from SuperNet.
            dst_conv: Destination Conv2d in subnet.
            c_in_sub: Number of input channels for subnet.
            c_out_sub: Number of output channels for subnet.
        """
        dst_conv.weight.copy_(src_conv.weight[:c_out_sub, :c_in_sub, :, :])
        if src_conv.bias is not None and dst_conv.bias is not None:
            dst_conv.bias.copy_(src_conv.bias[:c_out_sub])
    
    def _copy_bn(
        self,
        src_bn: nn.BatchNorm2d,
        dst_bn: nn.BatchNorm2d,
        c_sub: int
    ) -> None:
        """
        Copy BatchNorm parameters with channel slicing.
        
        Args:
            src_bn: Source BatchNorm2d from SuperNet.
            dst_bn: Destination BatchNorm2d in subnet.
            c_sub: Number of channels for subnet.
        """
        dst_bn.weight.copy_(src_bn.weight[:c_sub])
        dst_bn.bias.copy_(src_bn.bias[:c_sub])
        dst_bn.running_mean.copy_(src_bn.running_mean[:c_sub])
        dst_bn.running_var.copy_(src_bn.running_var[:c_sub])
        if src_bn.num_batches_tracked is not None:
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)
    
    def _copy_conv_bn(
        self,
        src_conv: nn.Conv2d,
        src_bn: nn.BatchNorm2d,
        dst_conv: nn.Conv2d,
        dst_bn: nn.BatchNorm2d,
        c_in_sub: int,
        c_out_sub: int
    ) -> None:
        """
        Copy Conv2d and BatchNorm2d weights with channel slicing.
        
        Args:
            src_conv: Source Conv2d.
            src_bn: Source BatchNorm2d.
            dst_conv: Destination Conv2d.
            dst_bn: Destination BatchNorm2d.
            c_in_sub: Input channels for subnet.
            c_out_sub: Output channels for subnet.
        """
        self._copy_conv(src_conv, dst_conv, c_in_sub, c_out_sub)
        self._copy_bn(src_bn, dst_bn, c_out_sub)
    
    def _copy_conv_bn_relu(
        self,
        src_conv: nn.Conv2d,
        src_bn: nn.BatchNorm2d,
        dst_conv: nn.Conv2d,
        dst_bn: nn.BatchNorm2d,
        c_in_sub: int,
        c_out_sub: int
    ) -> None:
        """
        Copy Conv-BN-ReLU block weights (ReLU has no weights).
        
        Args:
            src_conv: Source Conv2d.
            src_bn: Source BatchNorm2d.
            dst_conv: Destination Conv2d.
            dst_bn: Destination BatchNorm2d.
            c_in_sub: Input channels for subnet.
            c_out_sub: Output channels for subnet.
        """
        self._copy_conv_bn(src_conv, src_bn, dst_conv, dst_bn, c_in_sub, c_out_sub)
    
    def _copy_depthwise_conv(
        self,
        src_dw: nn.Conv2d,
        dst_dw: nn.Conv2d,
        c_sub: int,
        c_max: int
    ) -> None:
        """
        Copy depthwise convolution weights with channel slicing.
        
        Depthwise convolutions have groups=in_channels, so the weight tensor
        has shape (in_channels, 1, K, K).
        
        Args:
            src_dw: Source depthwise Conv2d.
            dst_dw: Destination depthwise Conv2d.
            c_sub: Number of channels for subnet.
            c_max: Number of channels for SuperNet.
        """
        dst_dw.weight.copy_(src_dw.weight[:c_sub, :, :, :])
        if src_dw.bias is not None and dst_dw.bias is not None:
            dst_dw.bias.copy_(src_dw.bias[:c_sub])
    
    def _copy_final_conv_weights(self, subnet: RadarNetwork, c_in_sub: int) -> None:
        """
        Copy final convolution weights from SuperNet to subnet.
        
        The final conv maps from decoder output channels to in_channels (output).
        
        Args:
            subnet: Target network to receive weights.
            c_in_sub: Input channels (last decoder's output channels).
        """
        src_conv = self.maximal_net.final_conv
        dst_conv = subnet.final_conv
        
        # Output channels = in_channels (fixed), input channels = last decoder output
        dst_conv.weight.copy_(src_conv.weight[:, :c_in_sub, :, :])
        if src_conv.bias is not None and dst_conv.bias is not None:
            dst_conv.bias.copy_(src_conv.bias)
    
    def sample_random_subnet(self) -> Tuple['RadarNode', RadarNetwork]:
        """
        Sample a random subnet architecture and return both the node and network.
        
        This is a convenience method for architecture search that randomly
        selects channel configurations and operations.
        
        Returns:
            Tuple of (RadarNode, RadarNetwork) with random architecture.
        """
        import random
        from search_spaces.radar_node import RadarNode
        
        # Create a new RadarNode
        node = RadarNode(
            in_channels=self.in_channels,
            initial_channels=self.initial_channels,
            channel_options=self.channel_options,
            num_encoder_stages=self.num_encoder_stages,
            num_nodes=self.num_nodes
        )
        
        # Randomly select actions until terminal
        while not node.is_terminal:
            actions = node.get_actions_tuples()
            if not actions:
                break
            action = random.choice(actions)
            node.play_action(action)
        
        # Sample subnet with weights
        subnet = self.sample(node)
        
        return node, subnet
    
    def get_num_parameters(self) -> int:
        """
        Return the total number of parameters in the SuperNet.
        
        Returns:
            Total parameter count.
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """
        Return the number of trainable parameters in the SuperNet.
        
        Returns:
            Trainable parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_selected_ops_from_node(self, node: 'RadarNode') -> Dict[str, str]:
        """
        Extract the selected operations from a RadarNode configuration.
        
        Args:
            node: A terminal RadarNode with all operations fixed.
            
        Returns:
            Dictionary mapping edge names to operation keys.
        """
        if not node.is_terminal:
            raise ValueError("RadarNode must be terminal (all operations fixed)")
        
        # RadarNode.get_cell_operations() now returns operation keys directly
        return node.get_cell_operations()
    
    def sample_random_architecture(self) -> 'RadarNode':
        """
        Sample a random architecture configuration without creating a subnet.
        
        This method is useful for training where we only need the operation
        selection, not an independent network copy.
        
        Returns:
            A terminal RadarNode with random architecture configuration.
        """
        import random
        from search_spaces.radar_node import RadarNode
        from types import SimpleNamespace
        config = {'supernet':
                    {'in_channels': self.in_channels,
                    'initial_channels': self.initial_channels,
                    'channel_options': self.channel_options,
                    'num_encoder_stages': self.num_encoder_stages,
                    'num_nodes': self.num_nodes}
        }
        # Create namespace for cleaner code
        def dict_to_namespace(d):
            """Recursively convert nested dicts to SimpleNamespace."""
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        cfg = dict_to_namespace(config)
        print(cfg.supernet.in_channels)
        node = RadarNode(
            cfg
        )
        
        while not node.is_terminal:
            actions = node.get_actions_tuples()
            if not actions:
                break
            action = random.choice(actions)
            node.play_action(action)
        
        return node


class SuperNetTrainer:
    """
    Utility class for training the SuperNet with path-wise sampling.
    
    During training, a random subnet path is sampled for each batch,
    and only that path's weights are updated. This implements the
    single-path one-shot training strategy.
    
    IMPORTANT: This trainer uses direct forward pass through the maximal_net
    with selected operations, ensuring gradients properly propagate back to
    the SuperNet weights. The sample() method creates independent weight
    copies and should only be used for evaluation/inference.
    
    Attributes:
        supernet: The SuperNet instance to train.
        optimizer: PyTorch optimizer for SuperNet parameters.
        criterion: Loss function.
    """
    
    def __init__(
        self,
        supernet: SuperNet,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ):
        """
        Initialise the trainer.
        
        Args:
            supernet: SuperNet instance to train.
            optimizer: Optimiser for SuperNet parameters.
            criterion: Loss function (e.g., nn.MSELoss for reconstruction).
        """
        self.supernet = supernet
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, 'RadarNode']:
        """
        Execute one training step with random path sampling.
        
        This method samples a random architecture and performs forward/backward
        passes directly through the SuperNet's maximal_net, ensuring gradients
        propagate correctly to update the shared weights.
        
        Args:
            inputs: Input tensor batch.
            targets: Target tensor batch.
            
        Returns:
            Tuple of (loss value, sampled RadarNode).
        """
        self.supernet.train()
        self.optimizer.zero_grad()
        
        # Sample a random architecture (just the configuration, not a subnet copy)
        node = self.supernet.sample_random_architecture()
        
        # Get the selected operations from the node
        selected_ops = self.supernet.get_selected_ops_from_node(node)
        
        # Forward pass directly through maximal_net with selected operations
        # This ensures gradients flow back to SuperNet weights
        outputs = self.supernet.maximal_net(inputs, selected_ops=selected_ops)
        loss = self.criterion(outputs, targets)
        
        # Backward pass - gradients now properly accumulate in SuperNet parameters
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), node
    
    def train_step_with_node(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        node: 'RadarNode'
    ) -> float:
        """
        Execute one training step with a specific architecture.
        
        Args:
            inputs: Input tensor batch.
            targets: Target tensor batch.
            node: A terminal RadarNode specifying the architecture to train.
            
        Returns:
            Loss value.
        """
        self.supernet.train()
        self.optimizer.zero_grad()
        
        selected_ops = self.supernet.get_selected_ops_from_node(node)
        outputs = self.supernet.maximal_net(inputs, selected_ops=selected_ops)
        loss = self.criterion(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()