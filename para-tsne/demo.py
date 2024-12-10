
import scipy.io
from train_par_tsne import train_par_tsne
import torch
torch.set_default_dtype(torch.float64)
from run_data_through_network import run_data_through_network
import matplotlib.pyplot as plt
traindata = scipy.io.loadmat("mnist_train.mat")
train_X = torch.from_numpy(traindata["train_X"])
train_labels = torch.from_numpy(traindata["train_labels"])

testdata = scipy.io.loadmat("mnist_test.mat")
test_X = torch.from_numpy(testdata["test_X"])
test_labels = torch.from_numpy(testdata["test_labels"])
del traindata, testdata

N = 300
chosen_tr = torch.randperm(60000)
chosen_tr = chosen_tr[:N]
train_X = train_X[chosen_tr,:]
train_labels = train_labels[chosen_tr]

perplexity = int(30)
layers = torch.tensor([500, 500, 2000, 2], dtype=torch.int32)

nets, err = train_par_tsne(train_X.clone(), train_labels, test_X, test_labels, layers)

mapped_train_X = run_data_through_network(nets, train_X)
mapped_train_X = mapped_train_X.numpy()
plt.scatter(mapped_train_X[:,0], mapped_train_X[:,1])
plt.show()