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
from braindecode.models import EEGNeX


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# helper View module (keeps same API as your repo's View)
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

def conv_output_size(size, kernel=4, stride=2, pad=1):
    return (size + 2*pad - kernel)//stride + 1

def conv_sequence_output(H, W):
    """
    Apply the encoder conv shapes in order:
      Conv2d(..., 4,2,1)  x4 times
      Conv2d(..., 4,1,0)  final
    Returns (H_out, W_out).
    """
    layers = [(4,2,1), (4,2,1), (4,2,1), (4,2,1), (4,1,0)]
    h, w = H, W
    for k, s, p in layers:
        h = (h + 2*p - k)//s + 1
        w = (w + 2*p - k)//s + 1
    return int(h), int(w)

def center_pad_or_crop_to(x, target_h, target_w):
    """
    Given tensor x with shape (B, C, Hx, Wx), return a tensor with shape (B, C, target_h, target_w)
    by center-cropping if larger or center-padding if smaller (pad constant 0).
    """
    B, C, Hx, Wx = x.shape
    # cropping
    if Hx > target_h:
        top = (Hx - target_h)//2
        x = x[:, :, top: top + target_h, :]
        Hx = target_h
    if Wx > target_w:
        left = (Wx - target_w)//2
        x = x[:, :, :, left: left + target_w]
        Wx = target_w

    # padding
    pad_top = max((target_h - Hx)//2, 0)
    pad_bottom = max(target_h - Hx - pad_top, 0)
    pad_left = max((target_w - Wx)//2, 0)
    pad_right = max(target_w - Wx - pad_left, 0)

    if any((pad_top, pad_bottom, pad_left, pad_right)):
        # F.pad expects pad = (left, right, top, bottom)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return x

# --- Encoder and Decoder (adapted) ---

class CELEBAEncoderAdapted(nn.Module):
    """
    Same architecture as your original CELEBAEncoder, adapted to:
      - single-channel input (no channel dim in x; accepts x shaped (B, H, W))
      - accepts x_shape=(H,W) in constructor to compute shapes for the Linear
      - returns (locs, scales) where scales are positive (same style as original)
    """
    def __init__(self, z_dim, x_shape):
        super().__init__()
        hidden_dim = 256
        self.z_dim = z_dim
        H, W = x_shape

        # same conv stack as original but in_channels=1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),   # same as original
            nn.ReLU(True),
            # We'll flatten to (B, hidden_dim * Hf * Wf) where Hf,Wf depend on x_shape
            # so we will not put a fixed View here in the Sequential (we'll do reshape after forward pass)
        )

        # compute encoder final spatial resolution
        Hf, Wf = conv_sequence_output(H, W)
        self._Hf = Hf
        self._Wf = Wf
        flattened = hidden_dim * Hf * Wf

        # linear to 2*z (mu and pre-scale)
        self.fc = nn.Linear(flattened, z_dim * 2)

        # keep these for compatibility (your repo had them)
        self.locs = nn.Linear(flattened, z_dim)
        self.scales = nn.Linear(flattened, z_dim)

        self.N = torch.distributions.Normal(0, 1)

        self.kl = 0

    def forward(self, x):
        self.N.loc = self.N.loc.cuda() #get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        
        # accept x as (B, H, W) or (B, 1, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # -> (B,1,H,W)
        hidden_map = self.encoder(x)           # (B, hidden_dim, Hf, Wf)
        B = hidden_map.size(0)
        hidden_flat = hidden_map.view(B, -1)  # (B, hidden_dim * Hf * Wf)
        out = self.fc(hidden_flat)            # (B, 2*z_dim)
        mu = out[:, :self.z_dim]
        pre_scale = out[:, self.z_dim:]
        # original code returned (locs, clamp(softplus(scales), min=1e-3))
        sigma = torch.clamp(F.softplus(pre_scale), min=1e-3)
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return mu, z

class CELEBADecoderAdapted(nn.Module):
    """
    Same architecture as original CELEBADecoder, adapted to:
      - accept x_shape=(H,W) to compute the starting spatial size for the
        linear->view block so that the conv-transpose stack mirrors encoder
      - produce single-channel output (no Sigmoid, unnormalized)
      - final output is cropped/padded to exactly (H,W)
    """
    def __init__(self, z_dim, x_shape):
        super().__init__()
        hidden_dim = 256
        H, W = x_shape

        # compute encoder final spatial resolution (to mirror exactly)
        Hf, Wf = conv_sequence_output(H, W)
        self._Hf = Hf
        self._Wf = Wf
        flattened = hidden_dim * Hf * Wf

        # linear to flattened features then view to (B, hidden_dim, Hf, Wf)
        self.fc = nn.Linear(z_dim, flattened)

        # same deconv stack as original
        self.deconv = nn.Sequential(
            # NOTE: we'll feed a tensor shaped (B, hidden_dim, Hf, Wf) into this
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),        # matches original
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),           # single-channel out; NO Sigmoid
            # output shape may not exactly match (H, W) for arbitrary inputs -> crop/pad in forward
        )

        self.output_shape = (H, W)
        self.hidden_dim = hidden_dim

    def forward(self, z):
        # z: (B, z_dim)
        B = z.size(0)
        h = self.fc(z)                                  # (B, flattened)
        h = h.view(B, self.hidden_dim, self._Hf, self._Wf)  # (B, hidden_dim, Hf, Wf)
        # apply the convtranspose stack
        x_recon = self.deconv(h)                        # (B, 1, Hx, Wx) maybe not exactly (H,W)
        # Ensure output exactly matches target spatial size
        target_h, target_w = self.output_shape
        x_recon = center_pad_or_crop_to(x_recon, target_h, target_w)
        # return shape (B, H, W) or (B,1,H,W)? keep single-channel 2D style consistent with your encoder:
        self.x_hat = x_recon.squeeze(1)
        return x_recon.squeeze(1)  # returns (B, H, W)

    
class HiggsVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, x_shape, pred_dim):
        super(HiggsVariationalAutoencoder, self).__init__()
        self.encoder = CELEBAEncoderAdapted(latent_dims, x_shape)
        self.decoder = CELEBADecoderAdapted(latent_dims, x_shape)
        self.pred_head = nn.Linear(latent_dims, pred_dim)


    def forward(self, x):
        mu_z, z = self.encoder(x)
        pred = self.pred_head(mu_z)
        x_hat = self.decoder(z)
        return pred
    
def train(model, data, device, epochs=20):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device) # GPU
            y = y.to(device)
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
    all_z = []
    all_y = []
    for i, (x, y) in enumerate(data):
        z = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        all_z.append(z)
        all_y.append(y)
    all_z = np.concatenate(all_z)
    all_y = np.concatenate(all_y)

    if z.shape[-1] > 2:
        all_z = UMAP(n_components=2, init='random', random_state=0).fit_transform(all_z)
    plt.scatter(all_z[:, 0], all_z[:, 1], c=all_y, cmap='tab10')
    plt.colorbar()
    plt.title("Latent space (projection) colored by regression target value")


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
            y = y.to(device)
            pred = model(x)
            mse += F.mse_loss(pred, y, reduction='sum')
            n_samples += pred.shape[0]
            
    mse /= n_samples
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
            x, _ = self.data[idx]      # (1, 28, 28)
            x = x.squeeze(0)           # (28, 28)
            y = x.mean().unsqueeze(0)  # (1,)
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