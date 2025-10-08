#%%
from typing import Optional, Callable
import torch
from geoopt.manifolds import Manifold
from functools import lru_cache, partial

class SPDManifold(Manifold):
    __scaling__ = Manifold.__scaling__.copy()
    ndim = 2  # SPD matrices are 2D (n x n)
    name = "SPD"
    
    def __init__(self, min_eigenval=1e-6):
        self.min_eigenval = min_eigenval
        super().__init__()
    
    def _check_point_on_manifold(self, point):
        """
        Check if point belongs to SPD manifold (batched).
        """
        is_symmetric = torch.allclose(point, point.transpose(-1, -2), atol=1e-5)
        if not is_symmetric:
            print("Matrix is not symmetric")
            sym_diff = (point - point.transpose(-1, -2)).abs()
            max_diff_per_matrix = sym_diff.max(dim=-1)[0].max(dim=-1)[0]
            print("max symm diff per mat: ",max_diff_per_matrix)

            return False
        eigvals = torch.linalg.eigvalsh(point)
        is_pos_def = torch.all(eigvals > 0, dim=-1)
        return torch.all(is_pos_def)

    def _check_vector_on_tangent(self, point, tol=1e-10):
        """
        Check if matrix is symmetric (batched)
        """
        return torch.allclose(point, point.transpose(-1, -2), atol=tol)

    def _ensure_symmetry(self, A):
        A_sym = 0.5 * (A + A.transpose(-1, -2))
        return A_sym
    
    def _trace(self, A):
        return torch.diagonal(A, dim1=-2, dim2=-1).sum(-1, keepdim=False)
    
    #mandatory geoopt
    @lru_cache(None)
    def _sym_funcm_impl(self, func, **kwargs):
        func = partial(func, **kwargs)

        def _impl(x):
            e, v = torch.linalg.eigh(x, "U")
            clipped_e = torch.clamp(e, min=self.min_eigenval)
            return v @ torch.diag_embed(func(clipped_e)) @ v.transpose(-1, -2)

        return _impl

    def _matrix_sqrt(self, A):
        A = self._ensure_symmetry(A)
        eigvals, eigvecs = torch.linalg.eigh(A)
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=self.min_eigenval))
        return self._ensure_symmetry(eigvecs @ torch.diag_embed(sqrt_eigvals) @ eigvecs.transpose(-1, -2))

    def _matrix_invsqrt(self, A):
        A = self._ensure_symmetry(A)
        eigvals, eigvecs = torch.linalg.eigh(A)
        inv_sqrt_eigvals = 1.0 / torch.sqrt(torch.clamp(eigvals, min=self.min_eigenval))
        return self._ensure_symmetry(eigvecs @ torch.diag_embed(inv_sqrt_eigvals) @ eigvecs.transpose(-1, -2))

    def _matrix_sqrt_and_invsqrt(self, A):
        A = self._ensure_symmetry(A)
        eigvals, eigvecs = torch.linalg.eigh(A)
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=self.min_eigenval))
        inv_sqrt_eigvals = 1.0 / sqrt_eigvals
        sqrt_A = eigvecs @ torch.diag_embed(sqrt_eigvals) @ eigvecs.transpose(-1, -2)
        invsqrt_A = eigvecs @ torch.diag_embed(inv_sqrt_eigvals) @ eigvecs.transpose(-1, -2)
        return self._ensure_symmetry(sqrt_A), self._ensure_symmetry(invsqrt_A)

    def _matrix_log(self, A):
        A = self._ensure_symmetry(A)
        eigvals, eigvecs = torch.linalg.eigh(A)
        log_eigvals = torch.log(torch.clamp(eigvals, min=self.min_eigenval))
        return self._ensure_symmetry(eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2))

    def _matrix_exp(self, A):
        A = self._ensure_symmetry(A)
        eigvals, eigvecs = torch.linalg.eigh(A)
        # eigvals = torch.clamp(eigvals, min = self.min_eigenval)
        exp_eigvals = torch.exp(eigvals)
        return self._ensure_symmetry(eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-1, -2))
    
    def _matrix_power(self, A, exponent):
        A = self._ensure_symmetry(A)
        eigvals, eigvecs = torch.linalg.eigh(A)
        eigvals = torch.clamp(eigvals, min = self.min_eigenval)
        exp_eigvals = torch.pow(eigvals, exponent)
        return self._ensure_symmetry(eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-1, -2))
    
    def _gen_eigenval(self, A, B):
        """
        Generalized Eigenvalue decomposition for SPD matrices only (batched)
        """
        assert self._check_point_on_manifold(A), "A does not belong to SPD manifold"
        assert self._check_point_on_manifold(B), "B does not belong to SPD manifold"

        L = torch.linalg.cholesky(B)
        L_inv = torch.linalg.inv(L)
        C = L_inv @ A @ L_inv.transpose(-1, -2)
        eigvals, _ = torch.linalg.eigh(C)
        return eigvals
    
    #mandatory geoopt
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.proju(x, u) @ x.transpose(-1, -2)
    
    #mandatory geoopt
    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        keepdim=False,
    ) -> torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)

        Raises
        ------
        ValueError
            if `keepdim` sine `torch.trace` doesn't support keepdim
        """
        if v is None:
            v = u
        inv_x = torch.matrix_power(x, -1)
        ret = self._trace(inv_x @ u @ inv_x @ v)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret
    
    #mandatory geoopt
    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = torch.matrix_power(x, -1)
        return self._ensure_symmetry(x + u + 0.5 * u @ inv_x @ u)
    
    #mandatory geoopt
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self._ensure_symmetry(u)
    
    #mandatory geoopt
    def sym_funcm(self,
        x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Apply function to symmetric matrix.

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix
        func : Callable[[torch.Tensor], torch.Tensor]
            function to apply

        Returns
        -------
        torch.Tensor
            symmetric matrix with function applied to
        """
        return self._sym_funcm_impl(func)(x)
    
    #mandatory goopt
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        symx = self._ensure_symmetry(x)
        return self.sym_funcm(symx, torch.abs)
        
    def norm(self, basepoint, V):
        """
        Norm of tangent vectors under affine-invariant metric
        basepoint: (n,n) SPD matrix (base point)
        V: (B,n,n) batch of tangent vectors (symmetric)
        """

        assert self._check_point_on_manifold(basepoint), "Basepoint should belong to SPD Manifold"
        assert self._check_vector_on_tangent(V), "All matrices should belong to SPF tangent space"

        bp_inv = torch.linalg.inv(basepoint)  
        return torch.norm(torch.matmul(bp_inv, V), dim=(1,2))
    
    def expmap(self, basepoint, point):
        """
        Exponential map according to affine-invariant metric (batched)
        """
        assert self._check_point_on_manifold(basepoint), "Basepoint not in SPD manifold"
        assert self._check_vector_on_tangent(point), "Point not in tangent space"

        bp_sqrt, bp_invsqrt = self._matrix_sqrt_and_invsqrt(basepoint)
        inner = bp_invsqrt @ point @ bp_invsqrt
        exp_inner = self._matrix_exp(inner)
        return self._ensure_symmetry(bp_sqrt @ exp_inner @ bp_sqrt)

    def log_map(self, basepoint, point):
        """
        Log map according to affine-invariant metric (batched)
        """
        assert self._check_point_on_manifold(basepoint), "Basepoint not in SPD manifold"
        assert self._check_point_on_manifold(point), "Point not in SPD manifold"

        bp_sqrt, bp_invsqrt = self._matrix_sqrt_and_invsqrt(basepoint)
        inner = bp_invsqrt @ point @ bp_invsqrt
        log_inner = self._matrix_log(inner)
        return self._ensure_symmetry(bp_sqrt @ log_inner @ bp_sqrt)

    def squared_distance(self, A, B):
        """
        Affine-Invariant Distance between SPD matrices (batched)
        """

        assert self._check_point_on_manifold(A), "A not in SPD manifold"
        assert self._check_point_on_manifold(B), "B not in SPD manifold"

        A_inv_sqrt = self._matrix_invsqrt(A)
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        log_inner = self._matrix_log(inner)
        return torch.linalg.norm(log_inner, dim=(-2, -1)).pow(2)

    def squared_distance_opt(self, A, B):
        """
        Optimized Affine-Invariant Distance (batched)
        """
        assert self._check_point_on_manifold(A), "A not in SPD manifold"
        assert self._check_point_on_manifold(B), "B not in SPD manifold"

        eigvals = self._gen_eigenval(A, B)
        eigvals = torch.clamp(eigvals, min=self.min_eigenval)
        return torch.sum(torch.log(eigvals).pow(2), dim=-1)

    def log_euclidean_mean(self, X):
        """
        Computes the Log-Euclidean Fréchet Mean (batched)
        """
        assert self._check_point_on_manifold(X), "Some X not in SPD manifold"
        mean_log = self._matrix_log(X).mean(dim=0)
        return self._matrix_exp(mean_log)

    def karcher_flow(self, X, steps=1, initialize="EucMean", eta=1, eps = 1e-6):
        """
        Computes Karcher flow for estimation of affine_invariant Fréchet mean (batched)
        """
        assert self._check_point_on_manifold(X), "Some X not in SPD manifold"

        if initialize == "EucMean":
            X_t = X.mean(dim=0)
        elif initialize == "Id":
            X_t = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)

        if steps == 1 and initialize == "Id" and eta == 1:
            return self.log_euclidean_mean(X)

        eps = 1e-6

        for s in range(steps):
            grad_t = self.log_map(X_t.expand(X.shape[0], -1, -1), X).mean(dim=0)
            grad_t = self._ensure_symmetry(grad_t)

            if torch.norm(grad_t) < eps:
                break
            X_t = self.expmap(X_t, eta * grad_t)
        return X_t
    
    def karcher_flow_improv(self, X, steps=1, initialize="EucMean", eta=1, eps = 1e-6):
        """
        Computes Karcher flow for estimation of affine_invariant Fréchet mean (batched)
        """
        assert self._check_point_on_manifold(X), "Some X not in SPD manifold"
            

        if initialize == "EucMean":
            X_t = X.mean(dim=0)
        elif initialize == "Id":
            X_t = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)

        bp_sqrt, bp_invsqrt = self._matrix_sqrt_and_invsqrt(X_t)
        
        if steps == 1 and initialize == "Id" and eta == 1:
            return self.log_euclidean_mean(X)

        for s in range(steps):
            bp_sqrt, bp_invsqrt = self._matrix_sqrt_and_invsqrt(X_t)
            grad_t = (self._matrix_log(bp_invsqrt.expand(X.shape[0], -1, -1) @ X @ bp_invsqrt.expand(X.shape[0], -1, -1))).mean(0)
            grad_t = self._ensure_symmetry(grad_t)
            
            print(f"grad norm: {torch.norm(grad_t)}")

            if torch.norm(grad_t) < eps:
                break
            X_t = bp_sqrt @ self._matrix_exp(eta *grad_t) @ bp_sqrt
        return X_t


    def interpolate(self, A, B, gamma=0.5):
        """
        Interpolates the affine-invariant geodesic path between A and B with a factor gamma 
        gamma==0: stays in A; 
        gamma==1: arrives at B; 
        gamma==0.5: goes to the geometric (affine-invariant) mean of A and B
        """
        assert self._check_point_on_manifold(A), "A does not belong to SPD manifold"
        assert self._check_point_on_manifold(B), "B does not belong to SPD manifold"

        A_sqrt, A_invsqrt = self._matrix_sqrt_and_invsqrt(A)
        inner = A_invsqrt @ B @ A_invsqrt
        log_inner = self._matrix_power(inner, gamma)
        return self._ensure_symmetry(A_sqrt @ log_inner @ A_sqrt)
    
    def parallel_transp_to_id(self, fromA, transportX):
        """
        Affine-invariant parallel transport X from A -> Id 
        (for both transportX on SPD manifold or on its tangent space)
        """
        assert (self._check_point_on_manifold(fromA) and self._check_point_on_manifold(transportX)) or \
               (self._check_point_on_manifold(fromA) and self._check_vector_on_tangent(transportX)), \
               "A doesn't belong to SPD manifold and/or X is neither on SPD manifold nor in it's tangent space"
        E = self._matrix_invsqrt(fromA)
        return self._ensure_symmetry(E.transpose(-1,-2) @ transportX @ E)
