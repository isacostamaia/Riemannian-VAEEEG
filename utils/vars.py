from abc import abstractmethod
import numpy as np
import torch
from models.modules.spdbn import SPDBatchNorm
from utils.manifold import SPDManifold


class Var():
    def __init__(self, manifold, distribution, shape, vec_shape, center_at, batchNorm):
        self.manifold = manifold
        self.distribution = distribution
        self.shape = shape
        self.vec_shape = vec_shape
        self.center_at = center_at
        self.batchNorm = batchNorm

    @abstractmethod
    def vectorize(self):
        pass
    @abstractmethod
    def unvectorize(self):
        pass

class SPDVar(Var):

    def __init__(self, distribution, shape, center_at, min_eigenval = 1e-6):
        """
            class of SPD observatio variables
            shape: n dimension of nxn matrices
        """
        self.min_eigenval = min_eigenval
        assert shape[-1]==shape[-2], "SPD matrices should be squared."
        vec_shape = int(shape[-1]*(shape[-1]+1)/2) 

        manifold = SPDManifold(min_eigenval=self.min_eigenval)
        self._triu_indices_cache = {} #indices for vectorization (matrix to vec and vice versa)

        super().__init__(manifold, distribution, shape, vec_shape, center_at, SPDBatchNorm)

    def vectorize(self, X):
        """
        X: (batch_size, n, n), symmetric matrices
        Returns: (batch_size, n*(n+1)//2), vectorized upper triangles
        """
        assert X.shape[1] == X.shape[2], "Input must be square matrices"
        
        n = self.shape[-1]
        device = X.device

        #Get upper triangle indices
        triu_indices = self.get_triu_indices(n, device)

        #Multiply upper-triang off-diagonql by sqrt(2) to keep module
        off_idxes = (triu_indices[0] != triu_indices[1])  # indices of upper-triangle off-diagonal elements
        X = X.clone()  # avoid modifying the original tensor
        X[:, triu_indices[0][off_idxes], triu_indices[1][off_idxes]] *= np.sqrt(2)

        #Extract upper triangle elements for all matrices in batch
        Y = X[:, triu_indices[0], triu_indices[1]]  # shape: (batch_size, n*(n+1)//2)

        return Y
    
    def unvectorize(self, vecs):
        """
        vecs: (batch_size, n*(n+1)//2), vectorized upper triangle
        Returns: (batch_size, n, n), symmetric matrices
        """

        batch_size, tri_size = vecs.shape

        n = self.shape[-1]
        device = vecs.device

        # Get upper triangle indices
        triu_indices = self.get_triu_indices(n, device)

        # Create empty symmetric matrix batch
        Y = torch.zeros((batch_size, n, n), device=device, dtype=vecs.dtype)

        # Fill upper triangle
        Y[:, triu_indices[0], triu_indices[1]] = vecs

        #Divide off-diagonal elements by sqrt(2) to restore original values
        off_idxes = (triu_indices[0] != triu_indices[1]) # indices of upper-triangle off-diagonal elements
        Y[:, triu_indices[0][off_idxes], triu_indices[1][off_idxes]] /= np.sqrt(2)

        # Reflect to lower triangle
        Y = Y + Y.transpose(1, 2)
        Y[:, range(n), range(n)] /= 2  # fix diagonal (added twice)
        assert (Y.transpose(1,2) == Y).all(), "Unvectorization failed: resulting matrix is not symmetric"

        return Y

    def get_triu_indices(self, n, device):
        if (n, device) not in self._triu_indices_cache:
            self._triu_indices_cache[(n, device)] = torch.triu_indices(n, n, device=device)
        return self._triu_indices_cache[(n, device)]
    

class EucVecVar(Var):
    def __init__(self, distribution, shape, center_at, batchNorm):
        manifold = "Euclidean"
        self.distribution = distribution
        self.shape = shape #one-dimensional
        self.center_at = center_at
        self.batchNorm = batchNorm

        assert type(shape)==int, "Eucliean Vector should have a single dimension."
        self._triu_indices_cache = {} #indices for vectorization (matrix to vec and vice versa)

        super().__init__(manifold=manifold, distribution=distribution,
                        shape=shape, vec_shape=shape, center_at=center_at, batchNorm=batchNorm)

    def vectorize(self, X):
        return X
    def unvectorize(self, vecs):
        return vecs