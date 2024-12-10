import torch
def run_data_through_network(nets, X):
    n = X.size(dim=0)
    mappedX = torch.cat([X, torch.ones(n,1)], dim=1)
    no_layers = len(nets)
    for i in range(no_layers-1):
        mappedX = torch.cat([1/(1+torch.exp( -torch.mm(mappedX, torch.cat([nets[i]['W'], nets[i]['bias_upW']],dim=0)) )), torch.ones(n,1)], dim=1)
    mappedX = torch.mm(mappedX, torch.cat([nets[-1]['W'], nets[-1]['bias_upW']], dim=0))
    return mappedX