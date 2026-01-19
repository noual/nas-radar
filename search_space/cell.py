import torch
import torch.nn as nn
from nni.nas.nn.pytorch import ModelSpace, LayerChoice
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt

from search_space.operations import Conv1x1, Conv1x3_3x1, Conv1x5_5x1, DilatedConv3x3_r2, DilatedConv3x3_r4, Identity, Zero

class RadarCell(ModelSpace):

    def __init__(self, C_in, C_out, num_nodes, label=None):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.num_nodes = num_nodes
        self.layers = nn.ModuleDict()

        for i in range(1, num_nodes):
            node_ops = nn.ModuleDict()
            for j in range(i):
                edge_name = f"edge_{j}_{i}"
                inp = C_in if j == 0 else C_out
                op_choices = OrderedDict(self.get_op_candidates(inp, C_out))
                node_ops[edge_name] = LayerChoice(op_choices, label=edge_name)
            self.layers[f"node_{i}"] = node_ops

    
    def get_op_candidates(self, in_channels, out_channels):
        """Returns a dictionary of operation candidates for LayerChoice"""
        return {
            'conv_1x1': Conv1x1(in_channels, out_channels),
            'conv_1x3_3x1': Conv1x3_3x1(in_channels, out_channels),
            'conv_1x5_5x1': Conv1x5_5x1(in_channels, out_channels),
            'dilated_conv_3x3_r2': DilatedConv3x3_r2(in_channels, out_channels),
            'dilated_conv_3x3_r4': DilatedConv3x3_r4(in_channels, out_channels),
            'identity': Identity(in_channels, out_channels),
            'none': Zero(in_channels, out_channels),
        }

    @property
    def is_terminal(self):
        for name, module in self.named_modules():
            if isinstance(module, LayerChoice):
                return False
        return True
    
    def get_actions(self):
        actions = {}
        for i, (name, module) in enumerate(self.layers.items()):
            for edge_name, node in module.items():
                if isinstance(node, LayerChoice):
                    actions[name+":"+node.label] = list(node.candidates.keys())
            if len(actions.keys()) > 0:
                return [(k, e) for k, v in actions.items() for e in v]
        return [(k, e) for k, v in actions.items() for e in v]
    
    def play_action(self, action: tuple):
        layer, edge = action[0].split(":")
        assert layer in self.layers, f"Invalid action key: {action[0]}"
        self.layers[layer][edge] = self.layers[layer][edge].candidates[action[1]]

    def forward(self, x):
        node_outputs = [x]
        for i in range(1, self.num_nodes):
            node_sum = 0
            for j in range(i):
                edge_name = f"edge_{j}_{i}"
                op = self.layers[f"node_{i}"][edge_name]
                node_sum = node_sum + op(node_outputs[j])
            node_outputs.append(node_sum)
        return node_outputs[-1]
    
    def draw(self):

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        for i in range(self.num_nodes):
            G.add_node(i)

        # Add edges with labels
        edge_labels = {}
        for i in range(1, self.num_nodes):
            for j in range(i):
                edge_name = f"edge_{j}_{i}"
                op = self.layers[f"node_{i}"][edge_name]
                
                # Get operation name
                op_name = op.__class__.__name__
                if hasattr(op, 'label'):
                    op_name = f"LayerChoice({op.label})"
                
                # Skip Zero operations for cleaner visualization
                if not isinstance(op, Zero):
                    G.add_edge(j, i)
                    edge_labels[(j, i)] = op_name

        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(2*self.num_nodes, 5))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=12, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        plt.title(f"RadarCell DAG (Nodes: {self.num_nodes}, C_in: {self.C_in}, C_out: {self.C_out})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def prune_cell(cell: RadarCell):
    """
    Prune a RadarCell by removing inactive nodes (nodes where all incoming edges are 'none').
    Returns a new, lighter RadarCell that is topologically equivalent.
    """
    # Identify which nodes are active (have at least one non-'none' incoming edge)
    active_nodes = [0]  # Node 0 (input) is always active
    
    for i in range(1, cell.num_nodes):
        has_active_input = False
        for j in range(i):
            edge_name = f"edge_{j}_{i}"
            op = cell.layers[f"node_{i}"][edge_name]
            # Check if this edge is not a Zero operation
            if not isinstance(op, Zero):
                has_active_input = True
                break
        if has_active_input:
            active_nodes.append(i)
        
    # If all nodes are active, return the original cell
    if len(active_nodes) == cell.num_nodes:
        return cell
    
    # Create a new cell with fewer nodes
    new_num_nodes = len(active_nodes)
    new_cell = RadarCell(cell.C_in, cell.C_out, new_num_nodes, cell._scope.name if hasattr(cell, '_scope') else None)
    
    # Map old node indices to new node indices
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(active_nodes)}
    
    # Copy operations from old cell to new cell, adjusting indices
    for new_i in range(1, new_num_nodes):
        old_i = active_nodes[new_i]
        new_j = 0
        for old_j in range(old_i):
            if old_j in active_nodes:
                old_edge_name = f"edge_{old_j}_{old_i}"
                new_edge_name = f"edge_{new_j}_{new_i}"
                # Copy the operation
                new_cell.layers[f"node_{new_i}"][new_edge_name] = cell.layers[f"node_{old_i}"][old_edge_name]
                new_j += 1
    
    return new_cell