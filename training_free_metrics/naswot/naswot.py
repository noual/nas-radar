import numpy as np
import torch
from torch.utils.data import DataLoader


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)

    return ld

class NASWOT:

    def __init__(self, data_loader):
        self.data_loader = DataLoader(data_loader, batch_size=8, shuffle=False)
        self.inputs, self.targets = next(iter(self.data_loader))
        self.inputs = self.inputs.to("cuda")
        self.targets = self.targets.to("cuda")

    def score(self, net):
        batch_size = len(self.targets)

        def counting_forward_hook(module, inp, out):
            inp = inp[0].view(inp[0].size(0), -1)
            x = (inp > 0).float()  # binary indicator
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance

        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        net.K = np.zeros((batch_size, batch_size))
        for name, module in net.named_modules():
            module_type = str(type(module))
            if ('ReLU' in module_type) and ('naslib' not in module_type):
                # module.register_full_backward_hook(counting_backward_hook)
                module.register_forward_hook(counting_forward_hook)

        x = torch.clone(self.inputs)
        net(x)
        s, jc = np.linalg.slogdet(net.K)

        return np.clip(jc, -1e10, 1e10) / 100
    
if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")
    from search_space.radar_node import RadarNode

    node = RadarNode(in_channels=3, initial_channels=8, channel_options=[8, 16, 32, 64], num_encoder_stages=3, num_nodes=4)
    while not node.is_terminal:
        actions = node.get_actions_tuples()
        action = actions[0]  # Just pick the first action for testing
        node.play_action(action)
    
    model = node.network
    model.prune_cells()
    model.to("cuda:0")
    x = torch.randn(10, 3, 32, 32).to("cuda:0")  # Example input batch of size 10
    
    naswot_evaluator = NASWOT(data_loader=torch.utils.data.TensorDataset(x, torch.zeros(x.size(0))))
    naswot_score = naswot_evaluator.score(model)
    print("NASWOT Score:", naswot_score)