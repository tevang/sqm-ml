# https://github.com/lschmiddey/Autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn import preprocessing

def numpyToTensor(x, device):
    x_train = torch.from_numpy(x).to(device)
    return x_train

class DataBuilder(Dataset):
    def __init__(self, df, device):
        self.x = df
        self.x = numpyToTensor(self.x, device)
        self.len=self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index]
    def __len__(self):
        return self.len


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden=50, hidden2=12, latent_dim=2):

        # Encoder
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden)
        self.linear2 = nn.Linear(hidden, hidden2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden2)
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        #         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, hidden2)
        self.fc_bn4 = nn.BatchNorm1d(hidden2)

        #         # Decoder
        self.linear4 = nn.Linear(hidden2, hidden2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=hidden2)
        self.linear5 = nn.Linear(hidden2, hidden)
        self.lin_bn5 = nn.BatchNorm1d(num_features=hidden)
        self.linear6 = nn.Linear(hidden, input_dim)
        self.lin_bn6 = nn.BatchNorm1d(num_features=input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar

class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


class get_vae_embeddings():

    def __init__(self, df):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_set = DataBuilder(df, self.device)
        self.trainloader = DataLoader(dataset=self.data_set, batch_size=1024)
        self.D_in = self.data_set.x.shape[1]


        self.hidden1 = 50
        self.hidden2 = 10

        # train
        self.epochs = 1500
        self.log_interval = 50
        self.val_losses = []
        self.train_losses = []
        self.mu_output = []
        self.logvar_output = []

        self.model = Autoencoder(self.D_in, self.hidden1, self.hidden2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_mse = customLoss()

    def train_vae(self):

        for epoch in range(1, self.epochs + 1):
            self.train(epoch)

        self.model.eval()
        test_loss = 0
        # no_grad() bedeutet wir nehmen die vorher berechneten Gewichte und erneuern sie nicht
        with torch.no_grad():
            for i, data in enumerate(self.trainloader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)

        # get mebeddings
        with torch.no_grad():
            for i, (data) in enumerate(self.trainloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)

                mu_tensor = mu
                self.mu_output.append(mu_tensor)
                mu_result = torch.cat(self.mu_output, dim=0)

                logvar_tensor = logvar
                self.logvar_output.append(logvar_tensor)
                logvar_result = torch.cat(self.logvar_output, dim=0)

        return mu_result


    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.trainloader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_mse(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        if epoch % 200 == 0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(self.trainloader.dataset)))
            self.train_losses.append(train_loss / len(self.trainloader.dataset))

# print(x0.shape)
# mm = get_vae_embeddings(x0)
# mm.train_vae()