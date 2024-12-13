
import scipy.io
from train_par_tsne import train_par_tsne
import torch
torch.set_default_dtype(torch.float64)
import scipy.io

from run_data_through_network import run_data_through_network
traindata = scipy.io.loadmat("mnist_train.mat")
train_X = torch.from_numpy(traindata["train_X"])
train_labels = torch.from_numpy(traindata["train_labels"])

testdata = scipy.io.loadmat("mnist_test.mat")
test_X = torch.from_numpy(testdata["test_X"])
test_labels = torch.from_numpy(testdata["test_labels"])
del traindata, testdata

N = 10000
chosen_tr = torch.randperm(60000)
chosen_tr = chosen_tr[:N]
train_X = train_X[chosen_tr,:]
train_labels = train_labels[chosen_tr]

perplexity = int(30)
layers = torch.tensor([500, 500, 2000, 2], dtype=torch.int32)

nets, err = train_par_tsne(train_X.clone(), train_labels, test_X, test_labels, layers)

mapped_train_X = run_data_through_network(nets, train_X)
mapped_train_X = mapped_train_X.numpy()
scipy.io.savemat("res_tr10000.mat", {'mapped_train_X':mapped_train_X, 'train_labels':train_labels.numpy()})
mapped_test_X = run_data_through_network(nets, test_X)
mapped_test_X = mapped_test_X.numpy()
scipy.io.savemat("res_test.mat", {'mapped_test_X':mapped_test_X, 'test_labels':test_labels.numpy()})

import matplotlib.pyplot as plt
plt.figure()
for c in range(1,11):
    this_class_idx = (test_labels==c).squeeze(-1) # torch.Size([nc]) 而非 .Size([nc,1])
    plt.scatter(mapped_test_X[this_class_idx,0], mapped_test_X[this_class_idx,1], s=1)
plt.axis('equal')
plt.show()
plt.pause()
