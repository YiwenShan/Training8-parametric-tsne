import torch
def compute_recon_err(machine, X):
    n,v = X.shape
    hid = 1/( 1 + torch.exp( -torch.mm(X, machine['W']) - machine['bias_upW'].repeat((n,1)) ) ) # n*h
    rec = 1/( 1 + torch.exp( -torch.mm(hid, machine['W'].T) - machine['bias_downW'].repeat((n,1)) ) ) # n*v
    # err = sum(sum((X - rec) .^ 2)) ./ size(X, 1);
    err = sum(sum( (X - rec)**2 ))/n
    return err