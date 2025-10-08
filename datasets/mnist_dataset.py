#%%
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 


class CovarianceTransform:
    def __init__(self, mean, std, epsilon=1e-3):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, image):
        # Convert PIL image to tensor and normalize
        image = transforms.ToTensor()(image)
        image = (image - self.mean) / self.std

        # Remove the channel dimension and zero-center rows
        image = image.squeeze(0)  # shape: [28, 28]
        image = image - image.mean(dim=1, keepdim=True)

        # Compute covariance matrix
        cov = torch.matmul(image, image.T) / (image.shape[1] - 1)
        cov += self.epsilon * torch.eye(cov.shape[0])  # Regularization

        return cov

def get_mnist_dataloader(batch_size):
        
    #Data
        #Define a transform to convert images to tensors
        transform = transforms.ToTensor()

        #Load the MNIST training dataset
        train_dataset = MNIST(root='~/datasets', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=False)

        #Compute mean and std
        data_iter = iter(train_loader)
        images, _ = next(data_iter)
        mean = images.mean()
        std = images.std()    

        #Instantiate the transform with computed mean and std
        transform = CovarianceTransform(mean=mean, std=std)

        #Load the datasets with the new transform
        train_dataset = MNIST(root='~/datasets', train=True, transform=transform, download=True)
        test_dataset = MNIST(root='~/datasets', train=False, transform=transform, download=True)

        #Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)

        return (train_loader, test_loader)


# train_loader, test_loader = get_mnist_dataloader(batch_size=5)
# %%
