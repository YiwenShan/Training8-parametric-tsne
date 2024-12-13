import torch
import copy
from run_data_through_network import run_data_through_network

def Hbeta(Di, beta_i):
    Pi = torch.exp(-Di*beta_i) # 1*(N-1) [exp(-beta*||xi-xk||2^2)  for k=1...N,≠i]
    sumPi = torch.sum(Pi) # 1*1  \sum_{k=1,≠i}^{N} exp(-||xi-xk||_2^2 * beta)
    H = torch.log(sumPi) + beta_i*sum( torch.mul(Di,Pi) )/sumPi # 之前 忘写/sumPi
    Pi = Pi/sumPi # 1*(N-1)  [exp(-beta*||xi-xk||2^2)/sumP  forall k≠i]
    return H,Pi

def x2p(X, u=30, tol=1e-5):

    n = X.size(dim=0)
    P = torch.zeros((n,n))
    beta = torch.ones((n,1)) # 不能0初始化啊！！！！
    lnU = torch.log(torch.tensor(u)) # ln(u) type()=torch.Tensor

    print('Computing pairwise distances...')
    sum_X = torch.sum(X**2, dim=1).reshape(n,1) # n*1
    D = sum_X.repeat((1,n)) + sum_X.T.repeat((n,1)) - 2*torch.mm(X,X.T) # n*n

    print('Computing P-values...')
    # 对于第i个样本, 二分搜索beta_i 使得 H(Pi) = -\sum_{j=1}^{n} p_{j|i} ln(p_{j|i}) = ln(u)
    # 最终输出beta:n*1 和 对应的非对称矩阵P:n*N  P_{ji} = p_{j|i}. 
    for i in range(n):
        if (i+1)%500==0: # (i+1)%500 == 0
            print('Computed P-values '+str(i+1)+' of '+str(n)+' datapoints...')
        
        # Set minimum and maximum values for precision
        betamin = torch.tensor(-float('inf'))
        betamax = torch.tensor(float('inf'))

        # Compute the Gaussian kernel and entropy for the current precision
        chosen_col = torch.arange(n) != i
        Di = D[i, chosen_col] # 1*(n-1)
        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - lnU
        tries = 0
        while torch.abs(Hdiff)>tol and tries<50:
            if Hdiff > 0:
                betamin = beta[i].clone() # 深拷贝, betamin=beta[i] 会导致: 改变beta[i], betamin同时改变
                if torch.isinf(betamax): beta[i] *= 2
                else: beta[i] = (beta[i]+betamax)/2
            else: # H <= lnU
                betamax = beta[i].clone()
                if torch.isinf(betamin): beta[i] /= 2
                else: beta[i] = (beta[i]+betamin)/2
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - lnU
            tries += 1
        
        # Set the final row of P
        P[i, chosen_col] = thisP
    print('Mean value of sigma: '  +str(torch.mean( 1/torch.sqrt(beta) )) )
    print('Minimum value of sigma: '+str(torch.min( 1/torch.sqrt(beta) )) )
    print('Maximum value of sigma: '+str(torch.max( 1/torch.sqrt(beta) )) )
    return P # P:n*n  beta:n*1


def tsne_grad(x, X, P, nets, v):
    n = X.size(dim=0) # 这一批的样本数 (即:batch size)
    no_layers = len(nets)
    
    ii = 0
    for i in range(no_layers):
        rowW, colW = nets[i]['W'].shape
        nets[i]['W'] = x[ii:ii+rowW*colW].reshape((colW,rowW)).T.clone() # torch是行优先, 须加.T
        ii += (rowW*colW)
        nbias_upW = nets[i]['bias_upW'].size(dim=1) # 假设 torch.Size([1,h]), 而非Size([h])
        nets[i]['bias_upW'] = x[ii:ii+nbias_upW].reshape((1,nbias_upW)).clone()
        ii += nbias_upW
    
    # Run the data through the network
    activations = [] # activations[i]: n*h  第i-1个网络的输出数据  h:第i-1个网络的输入维度
    activations.append( torch.cat([X, torch.ones(n,1)], dim=1) )
    for i in range(1,no_layers):
        activations.append( torch.cat([1/( 1+torch.exp(-torch.mm(activations[i-1], torch.cat( [nets[i-1]['W'], nets[i-1]['bias_upW']],dim=0)) ) ), torch.ones(n,1)], dim=1) )
    activations.append( torch.mm(activations[no_layers-1], torch.cat([nets[no_layers-1]['W'], nets[no_layers-1]['bias_upW']], dim=0)) )

    # Compute the Q-values
    sum_act = torch.sum(activations[no_layers]**2, dim=1).reshape(n,1) # n*1 precomputation for pairwise distances
    num = 1 + (sum_act.repeat(1,n) + sum_act.T.repeat(n,1) - 2*torch.mm(activations[no_layers], activations[no_layers].T))/v # num_ij = 1+||yi-yj||2^2/v
    Q = num**(-(v+1)/2) # n*n  Q_ij=(1 + ||yi-yj||2^2)^{-(v+1)/2}
    Q = Q - torch.diag_embed(Q.diag()) # Q[i,i] = 0
    num = Q # num_{ij} = (1 + ||yi-yj||2^2/v)^{-(v+1)/2}
    Q = Q/torch.sum(Q) # normalize to get probabilities
    Q = torch.max( Q, 1e-16*torch.ones(n,n) )

    C = torch.sum( P*torch.log(P/(Q+1e-16)) ).unsqueeze(0) # *,/ 均为元素对位相乘or相除 .unsqueeze(0)让size=[1]而非[0]
    Ix = torch.zeros(activations[no_layers].size()) # n*2
    stiff = ( (2*v + 2)/v ) * (P-Q) * num # n*n对称 stiff_{ij} = 4(P_ij-Q_ij)(1+||yi-yj||2^2)^{-1} 
    for i in range(n):
        Ix[i,:] = torch.sum((activations[no_layers][i,:].repeat(n,1) - activations[no_layers])*stiff[:,i].reshape(n,1).repeat(1,activations[no_layers].size(dim=1)), dim=0)
        # Ix(i,:) = dC/d(yi) = \sum_{j=1}^{n} (yi - yj)*4*(P_ji - Q_ji)*(1+||yj-yi||2^2/v)^{-1} \in 1*2

    # Compute gradients
    dW = []
    db = []
    for i in range(no_layers-1,-1,-1): # no_layers-1 ... 0
        delta = torch.mm(activations[i].T, Ix) # (2001*n)*(n*2) -> 
        dW.append( delta[:-1,:] )
        db.append( delta[ -1,:] )
        if i > 0:
            Ix = torch.mm(Ix, torch.cat([nets[i]['W'], nets[i]['bias_upW']], dim=0).T) * activations[i] * (1-activations[i])
            Ix = Ix[:,:-1]
    dW.reverse()
    db.reverse()

    # Convert gradient information
    dC = torch.zeros(x.size(dim=0),1)
    ii = 0
    for i in range(no_layers):
        dii = dW[i].numel()
        dC[ii:ii+dii,:] = dW[i].T.reshape(dii,1)
        ii += dii
        dii = db[i].numel()
        dC[ii:ii+dii,:] = db[i].T.reshape(dii,1)
        ii += dii

    return C, dC


def minimize(x, lens, X, P, nets, v):
    # import pdb; pdb.set_trace()
    RHO = torch.tensor(0.01)  # a bunch of constants for line searches
    SIG = torch.tensor(0.5)   # RHO and SIG are the constants in the Wolfe-Powell conditions
    INT = torch.tensor(0.1)   # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = torch.tensor(3.0)   # extrapolate maximum 3 times the current bracket
    MAX = torch.tensor(20)    # max 20 function evaluations per line search
    RATIO = torch.tensor(100) # maximum allowed slope ratio
    
    red = torch.tensor(1)
    if lens>0: S='Linesearch' 
    else: S='Function evaluation'
    
    i = 0 # zero the run length counter
    ls_failed = 0 # no previous line search has failed
    fX = torch.Tensor()
    f1, df1 = tsne_grad(x.clone(), X, P, copy.deepcopy(nets), v)
    i += (lens<0)
    s = -df1
    d1 = -torch.mm(s.T, s) # 1*1
    z1 = red/(1-d1)

    while i < abs(lens):
        i += (lens>0)
        x0 = x.clone()
        f0 = f1.clone()
        df0 = df1.clone() # make a copy of current values

        x += (z1*s) # begin line search x = x - z1*(dKL/dw)
        f2, df2 = tsne_grad(x.clone(), X, P, copy.deepcopy(nets), v)

        i += (lens<0)
        d2 = torch.mm(df2.T, s)

        f3 = f1.clone() # 不加.clone()也行(但二者地址一样,不知为啥能行). 因为f1=tnesor([x])而非tensor(x) <- 后者必须加.clone(), 例如MAX.clone()
        d3 = d1.clone()
        z3 = -z1 # initialize point 3 equal to point 1

        if lens>0: M=MAX.clone() # 若不加clone, 后面的M -= 1 会导致MAX 也-1
        else: M = torch.min(MAX, -lens-i)
        success = 0
        limit = -1
        while True:
            while (f2>f1+z1*RHO*d1) or (d2>-SIG*d1) and M>0:
                limit = z1.clone() # tighten the bracket
                if f2>f1:
                    z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3) # quadratic fit
                else:
                    A = 6*(f2-f3)/z3 + 3*(d2+d3) # cubic fit
                    B = 3*(f3-f2) - z3*(d3+2*d2)
                    z2 = (torch.sqrt(B*B - A*d2*z3*z3))/A
                if z2.isnan() or z2.isinf(): # 要求type(z2) = torch.Tensor()
                    z2 = z3/2 # if we had a numerical problem then bisect
                z2 = torch.max( torch.min(z2,INT*z3), (1-INT)*z3 ) # don't accept too close to limits
                z1 += z2 # updata the step (会导致limit跟着变)
                x = x + z2*s
                f2, df2 = tsne_grad(x.clone(), X, P, copy.deepcopy(nets), v)
                M -= 1 # BUG: 若M = MAX, 此处也会改变MAX
                i += (lens<0)
                d2 = torch.mm(df2.T, s)
                z3 -= z2
            if f2>f1+z1*RHO*d1 or d2>-SIG*d1: break
            elif d2 > SIG*d1:
                success = 1
                break
            elif M==0: break

            A = 6*(f2-f3)/z3 + 3*(d2+d3) # make cubic extrapolation
            B = 3*(f3-f2) - z3*(d3 + 2*d2)
            z2 = -d2*z3*z3/(B + torch.sqrt(B*B - A*d2*z3*z3)) # num. error possible - ok!

            if (not z2.isreal()) or z2.isnan() or z2.isinf() or z2<0:
                if limit < -0.5: z2 = z1*(EXT-1)
                else: z2 = (limit-z1)/2
            elif (limit > -0.5) and (z2+z1 > limit):
                z2 = (limit-z1)/2
            elif (limit < -0.5) and (z2+z1 > z1*EXT):
                z2 = z1*(EXT - 1.0)
            elif z2 < -z3*INT:
                z2 = -z3*INT
            elif (limit > -0.5) and (z2 < (limit-z1)*(1.0-INT)):
                z2 = (limit-z1)*(1.0-INT)
            
            f3 = f2.clone()
            d3 = d2.clone()
            z3 = -z2
            z1 += z2
            x += z2*s
            f2, df2 = tsne_grad(x.clone(), X, P, copy.deepcopy(nets), v)
            M -= 1
            i += (lens<0)
            d2 = torch.mm(df2.T, s)
            # end of this search
        
        if success:
            f1 = f2.clone()
            if fX.numel()==0: # fX为空张量
                fX = f1.T
            else: 
                # fX = torch.cat([fX.T, f1], dim=1).T
                fX = torch.cat([fX, f1.T], dim=0) # 与上式等效
            print('%s %d:  f1=%4.6e\r' % (S, i, f1))
            # Polack-Ribiere direction
            s = ( torch.mm(df2.T, df2) - torch.mm(df1.T, df2) )/( torch.mm(df1.T, df1) )*s - df2 # 一个数*s - df2

            tmp = df1.clone() # exchange df1 and df2
            df1 = df2.clone()
            df2 = tmp.clone()

            d2 = torch.mm(df1.T, s)
            if d2 > 0:
                s = -df1
                d2 = -torch.mm(s.T, s)
            z1 *= torch.min(RATIO, d1/(d2-1e-16))
            d1 = d2.clone()
            ls_failed = 0 # this line search did not fail
        else:
            x = x0.clone()
            f1 = f0.clone()
            print('%s %6d failed;  f1=%4.6e\r' % (S, i, f1))
            df1 = df0.clone()
            if ls_failed or i > abs(lens): 
                break
            tmp = df1.clone()
            df1 = df2.clone()
            df2 = tmp.clone()
            s = -df1
            d1 = -torch.mm(s.T, s)
            z1 = 1/(1-d1)
            ls_failed = 1 # this line search failed
        # end if
    # end while
    return x # Nw*1  Nw: 所有网络中的参数总数


def tsne_backprop(nets, train_X, train_labels, test_X, test_labels, max_iter=30, perplexity=30, v=1):

    n = train_X.size(dim=0) # n:样本数
    bs = min([5000,n]) # batch size
    ind = torch.randperm(n)
    err = torch.zeros((max_iter,1))

    # Precompute joint probabilities for all batches
    print('Precomputing P-values...')
    curX = []
    P = []
    i = 0
    for batch in range(0,n,bs): 
        ub = min([n, batch+bs-1])
        curX.append( train_X[ind[batch:ub+1],:].clone() ) # curX[i]: bs*v
        P.append( x2p(curX[i].clone(), perplexity, 1e-5).clone() ) # 求P:bs*bs (不对称版)
        P[i][P[i].isnan()] = 0
        P[i] = (P[i] + P[i].T)/2
        P[i] = P[i]/torch.sum(P[i]) # torch.sum(): 1*1 求总和
        P[i] = torch.max(P[i], torch.tensor(1e-16))
        i += 1

    no_layers = len(nets)
    # 梯度下降 开始迭代 min_{网络参数} KL(P||Q)
    for iter in range(max_iter):
        print('Iteration '+str(iter)+'...')
        b = 0
        for batch in range(0,n,bs):
            x = torch.Tensor() # 创建一空张量 存储所有参数
            for i in range(no_layers): # x尺寸: [Nw,1](列向量)  而非[Nw,](行向量) 
                if x.numel()==0:
                    x = nets[i]['W'].T.reshape((-1,1)).clone()
                else:
                    x = torch.cat([x, nets[i]['W'].T.reshape((-1,1)).clone()]) # matlab的W(:)是列优先地拉伸,故需.T再reshape
                x = torch.cat([x, nets[i]['bias_upW'].T.reshape((-1,1))]) 
            # Perform conjugate gradient using three linesearches
            x = minimize(x.clone(), 3, curX[b], P[b], copy.deepcopy(nets), v) # 深拷贝nets和nets
            b += 1

            # Store new solution
            no_layers = len(nets)
            ii = 0
            for i in range(no_layers):
                rowW, colW = nets[i]['W'].shape
                nets[i]['W'] = x[ii:ii+rowW*colW].reshape((colW,rowW)).T.clone() # torch是行优先, 须加.T
                ii += (rowW*colW)
                nbias_upW = nets[i]['bias_upW'].size(dim=1) # 假设 torch.Size([1,h]), 而非Size([h])
                nets[i]['bias_upW'] = x[ii:ii+nbias_upW].reshape((1,nbias_upW)).clone()
                ii += nbias_upW

        # Estimate the current error
        activations = run_data_through_network(nets, curX[0]) # bs*2
        sum_act = torch.sum(activations*activations, dim=1).reshape(bs,1) # bs*1
        Q = (1 + sum_act.repeat(1,bs) + sum_act.T.repeat(bs,1) - 2*torch.mm(activations, activations.T) )**(-(v+1)/2)
        Q = Q - torch.diag_embed(Q.diag()) # Q[i,i] = 0
        Q = Q/torch.sum(Q) # 
        Q = torch.max( Q, 1e-16*torch.ones(bs,bs) )
        KL = torch.sum( P[0]*torch.log(P[0]/(Q+1e-16)) ).unsqueeze(0) # *,/ 均为元素对位相乘or相除
        print('t-SNE error: KL(P||Q) = %f' % KL)

        err[iter] = onenn_error(run_data_through_network(nets,train_X), train_labels, \
                              run_data_through_network(nets, test_X), test_labels)
        print('1nn error: %f' % err[iter])
    return nets, err


def onenn_error(train_X, train_labels, test_X, test_labels, k=1):
    # Compute pairwise distance matrix
    n = train_X.size(dim=0) # 训练样本数
    ntest = test_X.size(dim=0) # 测试样本数
    sum_train = torch.sum(train_X*train_X, dim=1).reshape(n,1) # n*1
    sum_test = torch.sum(test_X*test_X, dim=1).reshape(ntest,1) # ntest*1
    D = sum_train.repeat(1,ntest) + sum_test.T.repeat(n,1) - 2*torch.mm(train_X, test_X.T) # n*ntest

    # labeling
    classification = torch.zeros(ntest,1)
    for j in range(ntest):
        sorted, idx = torch.sort(D[:,j]) # 第j个测试样本与n个训练样本的距离 升序排
        classification[j,0] = train_labels[idx[1]]
    # err = sum(test_labels ~= classification) ./ numel(test_labels);
    err = torch.sum(test_labels != classification)/ntest
    return err

