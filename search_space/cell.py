import torch
import torch.nn as nn
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt

from search_space.operations import Conv1x1, Conv1x3_3x1, Conv1x5_5x1, DilatedConv3x3_r2, DilatedConv3x3_r4, Identity, Zero, DepthwiseConv3x3, get_op_candidates


class RadarCell(nn.Module):
    """
    A DAG-based cell for Neural Architecture Search.
    
    Each cell represents a directed acyclic graph where nodes receive inputs
    from all preceding nodes via learnable operations (edges).
    
    Attributes:
        C_in: Number of input channels.
        C_out: Number of output channels.
        num_nodes: Number of nodes in the DAG (including input node 0).
        layers: ModuleDict containing operations for each edge.
    """

    def __init__(self, C_in, C_out, num_nodes, label=None, store_all_ops=False):
        """
        Initialise a RadarCell.
        
        Args:
            C_in: Number of input channels.
            C_out: Number of output channels.
            num_nodes: Number of nodes in the DAG.
            label: Optional label for the cell.
            store_all_ops: If True, edges contain all operations as a ModuleDict
                          (useful for SuperNet weight storage). If False, edges
                          start with all ops but can be fixed to single operations.
        """
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.num_nodes = num_nodes
        self._store_all_ops = store_all_ops
        self.layers = nn.ModuleDict()

        for i in range(1, num_nodes):
            node_ops = nn.ModuleDict()
            for j in range(i):
                edge_name = f"edge_{j}_{i}"
                inp = C_in if j == 0 else C_out
                op_choices = OrderedDict(self.get_op_candidates(inp, C_out))
                # Store all operations as ModuleDict
                node_ops[edge_name] = nn.ModuleDict(op_choices)
            self.layers[f"node_{i}"] = node_ops

    
    def get_op_candidates(self, in_channels, out_channels):
        """
        Return a dictionary of operation candidates.
        
        Args:
            in_channels: Number of input channels for the operation.
            out_channels: Number of output channels for the operation.
            
        Returns:
            Dictionary mapping operation names to instantiated modules.
        """
        return get_op_candidates(in_channels, out_channels)
    
    def get_operation_names(self):
        """
        Return the list of available operation names.
        
        Returns:
            List of operation name strings.
        """
        return list(get_op_candidates(1, 1).keys())
    
    def get_edge_operation(self, node_idx, edge_idx, op_name=None):
        """
        Retrieve an operation module from a specific edge.
        
        Args:
            node_idx: Index of the target node (1 to num_nodes-1).
            edge_idx: Index of the source node (0 to node_idx-1).
            op_name: If edge contains ModuleDict, specify which operation to
                    retrieve. If None and edge is ModuleDict, returns the
                    ModuleDict. If edge is a single operation, returns it.
                    
        Returns:
            The operation module at the specified edge.
        """
        edge_name = f"edge_{edge_idx}_{node_idx}"
        node_key = f"node_{node_idx}"
        edge_module = self.layers[node_key][edge_name]
        
        if isinstance(edge_module, nn.ModuleDict):
            # Edge contains all operations
            if op_name is not None:
                return edge_module[op_name]
            return edge_module
        else:
            # Edge is fixed to a specific operation
            return edge_module
    
    def set_edge_operation(self, node_idx, edge_idx, operation):
        """
        Set a specific operation at an edge (replaces ModuleDict with single op).
        
        Args:
            node_idx: Index of the target node.
            edge_idx: Index of the source node.
            operation: The operation module to set.
        """
        edge_name = f"edge_{edge_idx}_{node_idx}"
        node_key = f"node_{node_idx}"
        self.layers[node_key][edge_name] = operation
    
    def iterate_edges(self):
        """
        Iterate over all edges in the cell.
        
        Yields:
            Tuples of (node_idx, edge_idx, edge_name, edge_module).
        """
        for i in range(1, self.num_nodes):
            for j in range(i):
                edge_name = f"edge_{j}_{i}"
                yield i, j, edge_name, self.layers[f"node_{i}"][edge_name]

    @property
    def is_terminal(self):
        """Check if all edges have been fixed to single operations."""
        for node_idx, edge_idx, edge_name, edge_module in self.iterate_edges():
            if isinstance(edge_module, nn.ModuleDict):
                return False
        return True
    
    def get_actions(self):
        """Return available actions (unfixed edges with their operation choices)."""
        actions = {}
        for i, (name, module) in enumerate(self.layers.items()):
            for edge_name, node in module.items():
                if isinstance(node, nn.ModuleDict):
                    actions[name+":"+edge_name] = list(node.keys())
            if len(actions.keys()) > 0:
                return [(k, e) for k, v in actions.items() for e in v]
        return [(k, e) for k, v in actions.items() for e in v]
    
    def play_action(self, action: tuple):
        """Fix an edge to a specific operation."""
        layer, edge = action[0].split(":")
        assert layer in self.layers, f"Invalid action key: {action[0]}"
        edge_module = self.layers[layer][edge]
        if isinstance(edge_module, nn.ModuleDict):
            self.layers[layer][edge] = edge_module[action[1]]
        else:
            raise ValueError(f"Edge {edge} is already fixed to an operation")

    def forward(self, x, selected_ops=None):
        """
        Forward pass through the cell.
        
        Args:
            x: Input tensor.
            selected_ops: Optional dict mapping edge names to operation names.
                         Used in SuperNet mode to select which operations to use.
                         If None and edge is ModuleDict, sums all operations.
                         If None and edge is fixed, uses that operation.
                         
        Returns:
            Output tensor from the last node.
        """
        node_outputs = [x]
        for i in range(1, self.num_nodes):
            node_sum = 0
            for j in range(i):
                edge_name = f"edge_{j}_{i}"
                edge_module = self.layers[f"node_{i}"][edge_name]
                
                if selected_ops is not None and edge_name in selected_ops:
                    # Explicit operation selection
                    op_name = selected_ops[edge_name]
                    if isinstance(edge_module, nn.ModuleDict):
                        op = edge_module[op_name]
                    else:
                        op = edge_module
                    node_sum = node_sum + op(node_outputs[j])
                elif isinstance(edge_module, nn.ModuleDict):
                    # SuperNet mode without explicit selection: sum all ops
                    for op_name, op in edge_module.items():
                        node_sum = node_sum + op(node_outputs[j])
                else:
                    # Fixed operation
                    op = edge_module
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
                if isinstance(op, nn.ModuleDict):
                    op_name = f"[{len(op)} ops]"
                else:
                    op_name = op.__class__.__name__
                
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