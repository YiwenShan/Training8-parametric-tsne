
import torch
import copy
torch.set_default_dtype(torch.float64)
from tsne_backprop import tsne_backprop


def train_par_tsne(train_X, train_labels, test_X, test_labels, layers):
    
    oriX = train_X.clone() # type()=torch.Tensor
    N,D = train_X.shape
    no_layers = len(layers) # 网络数
    nets = [] # 存储每个网络  一个网络是一个字典, 一个字典有3个键值对, 值为torch.Tensor()

    for i in range(0,no_layers): # i=0,1,2,3
        print( 'Training layer '+str(i+1)+'(size '+str(D)+'->'+str(layers[i].item())+')...' )
        if i != no_layers-1:
            nets.append( train_rbm(train_X.clone(), layers[i]) ) 
            train_X = 1/( 1+torch.exp( -torch.mm(train_X, nets[i]['W']) - nets[i]['bias_upW'] ) ) # Transform data
        else:# i = no_layers-1
            # nets.append( train_lin_rbm(train_X.clone(), layers[i]) ) # linear boltzmann
            nets.append( train_lin(train_X.clone(), layers[i]) ) # fully connected layer

    # Perform backpropagation of the network using t-SNE gradient  # max_iter=30, perplexity=30, t分布自由度v=1
    nets, err = tsne_backprop(copy.deepcopy(nets), oriX, train_labels, test_X, test_labels, 30, 30, 1) 
    return nets, err


def train_rbm(X, h, eta=0.1, max_iter=int(30), weight_cost=2e-4):
    n,v = X.shape # n:样本数 v:特征数
    W = torch.normal(mean=0.0, std=0.1, size=(v,h)) # W ~ N(0, 0.01)
    bias_upW = torch.zeros((1,h))
    bias_downW = torch.zeros((1,v))

    dW = torch.zeros((v,h))
    dBias_upW = torch.zeros((1,h))
    dBias_downW = torch.zeros((1,v))

    bs = 100; # batch size
    ind = torch.randperm(n) # [0,n-1] dtype=torch.int64
    for iter in range(0,max_iter):
        if iter<5:momentum = 0.5
        else:momentum = 0.9
        for batch in range(0,n-bs,bs): # start, stop, step
            ub = min(batch+bs-1, n)
            vis1 = X[ind[batch:ub+1],:] # bs*v  ind.dtype必须为torch.int64. int32不行
            hid1 = 1/(1 + torch.exp( -torch.mm(vis1,W) - bias_upW.repeat((bs,1)) ) ) # bs*h

            hid_states = (hid1 > torch.rand((bs,h))).type(torch.float64) # bs*h  {0.0, 1.0}
            vis2 = 1/(1 + torch.exp( -torch.mm(hid_states,W.T) - bias_downW.repeat((bs,1)) ) ) # bs*v
            hid2 = 1/(1 + torch.exp( -torch.mm(vis2, W) - bias_upW.repeat((bs,1)) ) ) # bs*h

            posprods = torch.mm(vis1.T, hid1)/bs # (v*bs)*(bs*h) = v*h
            negprods = torch.mm(vis2.T, hid2)/bs # (v*bs)*(bs*h) = v*h
            dW = momentum*dW + eta*(posprods - negprods + weight_cost*W)
            dBias_upW = momentum*dBias_upW + eta/bs * (sum(hid1,0) - sum(hid2,0)) # 1*h
            dBias_downW = momentum*dBias_downW + eta/bs * (sum(vis1,0) - sum(vis2,0)) # 1*v

            W += dW
            bias_upW += dBias_upW
            bias_downW += dBias_downW
        machine = {'W':W, 'bias_upW':bias_upW, 'bias_downW':bias_downW}
        siz = min([n, 5000])
        err = compute_recon_err(machine, X[:siz,:])
        print('Iter '+str(iter)+'(rec. err = '+str(err)+')...')
    return machine


def compute_recon_err(machine, X):
    n,v = X.shape
    hid = 1/( 1 + torch.exp( -torch.mm(X, machine['W']) - machine['bias_upW'].repeat((n,1)) ) ) # n*h
    rec = 1/( 1 + torch.exp( -torch.mm(hid, machine['W'].T) - machine['bias_downW'].repeat((n,1)) ) ) # n*v
    err = sum(sum( (X - rec)**2 ))/n
    return err


def train_lin_rbm(X, h, eta=1e-3, max_iter=50, weight_cost=2e-4):
    n,v = X.shape # n:样本数 v:特征数
    W = torch.normal(mean=0.0, std=0.1, size=(v,h)) # W ~ N(0, 0.01)
    bias_upW = torch.zeros((1,h))
    bias_downW = torch.zeros((1,v))

    dW = torch.zeros((v,h))
    dBias_upW = torch.zeros((1,h))
    dBias_downW = torch.zeros((1,v))

    bs = 100; # batch size
    for iter in range(0,max_iter):
        err = 0
        ind = torch.randperm(n) # [0,n-1] dtype=torch.int64

        if iter<5: momentum = 0.5
        else: momentum = 0.9
        for batch in range(0,n-bs,bs):
            ub = min(batch+bs-1, n)
            vis1 = X[ind[batch:ub+1],:] # bs*v
            hid1 = torch.mm(vis1,W) + bias_upW.repeat((bs,1)) # bs*h

            vis2 = 1/(1 + torch.exp( -torch.mm(hid1,W.T) - bias_downW.repeat((bs,1)) ) ) # bs*v
            hid2 = torch.mm(vis2, W) + bias_upW.repeat((bs,1)) # bs*h

            posprods = torch.mm(vis1.T, hid1)/bs # (v*bs)*(bs*h) = v*h
            negprods = torch.mm(vis2.T, hid2)/bs # (v*bs)*(bs*h) = v*h
            dW = momentum*dW + eta*(posprods - negprods - weight_cost*W) # 忘加 -wieght_cost*W
            dBias_upW = momentum*dBias_upW + eta/bs * (sum(hid1,0) - sum(hid2,0)) # 1*h
            dBias_downW = momentum*dBias_downW + eta/bs * (sum(vis1,0) - sum(vis2,0)) # 1*v

            W += dW
            bias_upW += dBias_upW
            bias_downW += dBias_downW
            err += torch.sum( (vis1-vis2)**2 )/n
        print('Iter '+str(iter)+'(rec. err = '+str(err)+')...')
    machine = {'W':W, 'bias_upW':bias_upW, 'bias_downW':bias_downW}
    return machine


def train_lin(X, h, eta=1e-3, max_iter=50, weight_cost=2e-4):
    n,v = X.shape # n:样本数 v:特征数
    W = torch.normal(mean=0.0, std=0.1, size=(v,h)) # W ~ N(0, 0.01)
    bias_upW = torch.zeros((1,h))
    bias_downW = torch.zeros((1,v))

    dW = torch.zeros((v,h))
    dBias_upW = torch.zeros((1,h))
    dBias_downW = torch.zeros((1,v))

    bs = 100; # batch size
    for iter in range(0,max_iter):
        err = 0
        ind = torch.randperm(n) # [0,n-1] dtype=torch.int64

        for batch in range(0,n-bs,bs):
            ub = min(batch+bs-1, n)
            vis1 = X[ind[batch:ub+1],:] # bs*v
            hid1 = torch.mm(vis1,W) + bias_upW.repeat((bs,1)) # bs*h

            # hid_states = hid1 + torch.normal(mean=0.0, std=1.0, size=(bs,h)) # bs*h {True, False}
            vis2 = 1/(1 + torch.exp( -torch.mm(hid1,W.T) - bias_downW.repeat((bs,1)) ) ) # bs*v
            hid2 = torch.mm(vis2, W) + bias_upW.repeat((bs,1)) # bs*h

            posprods = torch.mm(vis1.T, hid1)/bs # (v*bs)*(bs*h) = v*h
            negprods = torch.mm(vis2.T, hid2)/bs # (v*bs)*(bs*h) = v*h
            dW = eta*(posprods - negprods - weight_cost*W) # 忘加 -wieght_cost*W
            dBias_upW = eta/bs * (sum(hid1,0) - sum(hid2,0)) # 1*h
            dBias_downW = eta/bs * (sum(vis1,0) - sum(vis2,0)) # 1*v

            W += dW
            bias_upW += dBias_upW
            bias_downW += dBias_downW
            err += torch.sum( (vis1-vis2)**2 )
        print('Iter '+str(iter)+'(rec. err = '+str(err/n)+')...')
    machine = {'W':W, 'bias_upW':bias_upW, 'bias_downW':bias_downW}
    return machine

