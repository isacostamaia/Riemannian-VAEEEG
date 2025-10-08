#%%
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from umap import UMAP

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, x_shape):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(x_shape[-1]*x_shape[-2], 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)

        self.kl = 0

    def forward(self, x):
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims, x_shape):
        super(Decoder, self).__init__()
        self.x_shape = x_shape
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, x_shape[-1]*x_shape[-2])
        self.x_hat = 0

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z)) #modify later when data is non-binary
        self.x_hat = z.reshape((-1, 1, self.x_shape[-2], self.x_shape[-1]))
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, x_shape, pred_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, x_shape)
        self.decoder = Decoder(latent_dims, x_shape)
        self.pred_head = nn.Linear(latent_dims, pred_dim)


    def forward(self, x):
        z = self.encoder(x)
        pred = self.pred_head(z)
        x_hat = self.decoder(z)
        return pred
    
def train(model, data, device, epochs=20):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device) # GPU
            y = y.to(device).unsqueeze(1)
            opt.zero_grad()
            pred = model(x)
            recon_loss = ((x - model.decoder.x_hat)**2).sum()
            # pred_loss = F.cross_entropy(pred, y, reduction='sum')
            pred_loss = F.mse_loss(pred, y, reduction='sum')
            loss = recon_loss + model.encoder.kl + pred_loss
            loss.backward()
            opt.step()

            # #batch acc
            # preds = pred.argmax(dim=-1)
            # bacc = (preds == y).sum()/preds.shape[0]

        print(f"btrain loss: {loss.item()}, brecon_loss: {recon_loss.item()}, bkl loss: {model.encoder.kl}, bpred_loss:{pred_loss.item()}")
    return model

def plot_latent(model, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        if z.shape[-1] > 2:
            z = UMAP(n_components=2, init='random', random_state=0).fit_transform(z)
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.title("regression target value")


def plot_reconstructed(model, r0=(-5, 10), r1=(-10, 5), n=12):
    #only for latent_dim =2
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = model.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

def evaluate(model, test_data, device):
    model.eval()
    mse = 0
    n_samples = 0
    with torch.no_grad():
        for x, y in test_data:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pred = model(x)
            mse += F.mse_loss(pred, y, reduction='sum')
            
    mse /= pred.shape[0]
    print(f"test MSE: {mse}")
    return mse



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    latent_dims = 2

    class MNISTRegression(torch.utils.data.Dataset):
        def __init__(self, train=True):
            self.data = torchvision.datasets.MNIST(
                './data',
                train=train,
                transform=torchvision.transforms.ToTensor(),
                download=True
            )
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            x, _ = self.data[idx]
            y = x.mean()  # average brightness
            return x, y

    data = torch.utils.data.DataLoader(MNISTRegression(train=True), batch_size=128, shuffle=True)
    test_data  = torch.utils.data.DataLoader(MNISTRegression(train=False), batch_size=128)

    n_targets = 1 #regression scalar     #len(data.dataset.targets.unique())
    x_shape = (28,28)
    vae = VariationalAutoencoder(latent_dims, x_shape, n_targets).to(device) # GPU
    vae = train(vae, data, device)
    plot_latent(vae, data)
    plt.show()
    plt.close()
    # plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
    test_acc = evaluate(vae, test_data, device)



# %%
#bckup unsupervised vae
#%%
# import torch; torch.manual_seed(0)
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils
# import torch.distributions
# import torchvision
# import numpy as np
# import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

# class VariationalEncoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalEncoder, self).__init__()
#         self.linear1 = nn.Linear(784, 512)
#         self.linear2 = nn.Linear(512, latent_dims)
#         self.linear3 = nn.Linear(512, latent_dims)

#         self.N = torch.distributions.Normal(0, 1)
#         self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
#         self.N.scale = self.N.scale.cuda()
#         self.kl = 0

#     def forward(self, x):
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.linear1(x))
#         mu =  self.linear2(x)
#         sigma = torch.exp(self.linear3(x))
#         z = mu + sigma*self.N.sample(mu.shape)
#         self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
#         return z
    
# class Decoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Decoder, self).__init__()
#         self.linear1 = nn.Linear(latent_dims, 512)
#         self.linear2 = nn.Linear(512, 784)

#     def forward(self, z):
#         z = F.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         return z.reshape((-1, 1, 28, 28))
    
# class VariationalAutoencoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder, self).__init__()
#         self.encoder = VariationalEncoder(latent_dims)
#         self.decoder = Decoder(latent_dims)

#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)
    
# def train(model, data, device, epochs=20):
#     opt = torch.optim.Adam(model.parameters())
#     for epoch in range(epochs):
#         for x, y in data:
#             x = x.to(device) # GPU
#             opt.zero_grad()
#             x_hat = model(x)
#             loss = ((x - x_hat)**2).sum() + model.encoder.kl
#             loss.backward()
#             opt.step()
#     return model

# def plot_latent(model, data, num_batches=100):
#     for i, (x, y) in enumerate(data):
#         z = model.encoder(x.to(device))
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break

# def plot_reconstructed(model, r0=(-5, 10), r1=(-10, 5), n=12):
#     w = 28
#     img = np.zeros((n*w, n*w))
#     for i, y in enumerate(np.linspace(*r1, n)):
#         for j, x in enumerate(np.linspace(*r0, n)):
#             z = torch.Tensor([[x, y]]).to(device)
#             x_hat = model.decoder(z)
#             x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
#             img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
#     plt.imshow(img, extent=[*r0, *r1])


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# latent_dims = 2

# data = torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('./data',
#                transform=torchvision.transforms.ToTensor(),
#                download=True),
#         batch_size=128,
#         shuffle=True)

# vae = VariationalAutoencoder(latent_dims).to(device) # GPU
# vae = train(vae, data, device)
# plot_latent(vae, data)
# plt.show()
# plt.close()
# plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))



#backup VAE with classifier pred head (supervised) good with latent_dim=4
# class VariationalEncoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalEncoder, self).__init__()
#         self.linear1 = nn.Linear(784, 512)
#         self.linear2 = nn.Linear(512, latent_dims)
#         self.linear3 = nn.Linear(512, latent_dims)

#         self.N = torch.distributions.Normal(0, 1)
#         self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
#         self.N.scale = self.N.scale.cuda()
#         self.kl = 0

#     def forward(self, x):
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.linear1(x))
#         mu =  self.linear2(x)
#         sigma = torch.exp(self.linear3(x))
#         z = mu + sigma*self.N.sample(mu.shape)
#         self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
#         return z
    
# class Decoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Decoder, self).__init__()
#         self.linear1 = nn.Linear(latent_dims, 512)
#         self.linear2 = nn.Linear(512, 784)

#     def forward(self, z):
#         z = F.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         return z.reshape((-1, 1, 28, 28))
    
# class VariationalAutoencoder(nn.Module):
#     def __init__(self, latent_dims, pred_dim):
#         super(VariationalAutoencoder, self).__init__()
#         self.encoder = VariationalEncoder(latent_dims)
#         self.decoder = Decoder(latent_dims)
#         self.pred_head = nn.Linear(latent_dims, pred_dim)


#     def forward(self, x):
#         z = self.encoder(x)
#         pred = self.pred_head(z)
#         return self.decoder(z), pred
    
# def train(model, data, device, epochs=20):
#     opt = torch.optim.Adam(model.parameters())
#     for epoch in range(epochs):
#         for x, y in data:
#             x = x.to(device) # GPU
#             y = y.to(device)
#             opt.zero_grad()
#             x_hat, pred = model(x)
#             recon_loss = ((x - x_hat)**2).sum()
#             pred_loss = F.cross_entropy(pred, y, reduction='sum')
#             loss = recon_loss + model.encoder.kl + pred_loss
#             loss.backward()
#             opt.step()

#             #batch acc
#             preds = pred.argmax(dim=-1)
#             bacc = (preds == y).sum()/preds.shape[0]

#         print(f"btrain loss: {loss.item()}, brecon_loss: {recon_loss.item()}, bkl loss: {model.encoder.kl}, bpred_loss:{pred_loss.item()}, bacc: {bacc}")
#     return model

# def plot_latent(model, data, num_batches=100):
#     for i, (x, y) in enumerate(data):
#         z = model.encoder(x.to(device))
#         z = z.to('cpu').detach().numpy()
#         if z.shape[-1] > 2:
#             z = UMAP(n_components=2, init='random', random_state=0).fit_transform(z)
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break

# def plot_reconstructed(model, r0=(-5, 10), r1=(-10, 5), n=12):
#     #only for latent_dim =2
#     w = 28
#     img = np.zeros((n*w, n*w))
#     for i, y in enumerate(np.linspace(*r1, n)):
#         for j, x in enumerate(np.linspace(*r0, n)):
#             z = torch.Tensor([[x, y]]).to(device)
#             x_hat = model.decoder(z)
#             x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
#             img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
#     plt.imshow(img, extent=[*r0, *r1])

# def evaluate(model, test_data, device):
#     model.eval()
#     correct = 0
#     n_samples = 0
#     with torch.no_grad():
#         for x, y in test_data:
#             x = x.to(device)
#             y = y.to(device)
#             x_hat, pred = model(x)
#             preds = pred.argmax(dim=-1)
#             correct += (preds==y).sum()
#             n_samples += y.shape[0]
#     acc = correct/n_samples
#     print(f"test acc: {acc}")
#     return acc


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# latent_dims = 2

# data = torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('./data',
#                transform=torchvision.transforms.ToTensor(),
#                download=True),
#         batch_size=128,
#         shuffle=True)
# test_data = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True),
#     batch_size=128,
#     shuffle=False
# )
# n_targets = len(data.dataset.targets.unique())
# vae = VariationalAutoencoder(latent_dims, n_targets).to(device) # GPU
# vae = train(vae, data, device)
# plot_latent(vae, data)
# plt.show()
# plt.close()
# # plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
# test_acc = evaluate(vae, test_data, device)