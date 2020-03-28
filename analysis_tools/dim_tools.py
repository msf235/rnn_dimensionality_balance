import numpy as np
from tempfile import mkdtemp
from joblib import Memory
import os

from scipy.spatial.distance import pdist, squareform


cache_dir = '../joblib_cache'
os.makedirs(cache_dir, exist_ok=True)

# cachedir = mkdtemp(dir=cache_dir)
memory = Memory(cachedir=cache_dir, verbose=1)


def svd_flip(u, vt, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u, vt : ndarray
        u and vt are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, vt)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of vt. The choice of which variable to base the
        decision on is generally algorithm dependent.


    Returns
    -------
    u_adjusted, vt_adjusted : arrays with the same dimensions as the input.

    """
    u = u.copy()
    vt = vt.copy()
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        vt *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(vt), axis=1)
        signs = np.sign(vt[range(vt.shape[0]), max_abs_rows])
        u *= signs
        vt *= signs[:, np.newaxis]
    return u, vt


# @memory.cache
def get_pcs(X, pcs, original_shape=True):
    """
    Return principal components of X.
    Args:
        X ([num_samples, ambient space dimension]): Data matrix of samples where each sample corresponds to a row of X.
        pcs ([num_pcs,]): List of principal components to return.

    Returns:
        pca_proj: ([len(pcs), num_samples]): Projection of X onto principal components given by pcs.

    """
    # X = X.copy()
    X_shape = X.shape
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
        print("Warning: concatenated first however many dimensions to get square data array")
    X_centered = X - np.mean(X, axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    # pca_proj = (s[pcs] * U[:, pcs]).T
    pca_proj = s[pcs] * U[:, pcs]
    if original_shape:
        pca_proj = pca_proj.reshape(*X_shape[:-1], pca_proj.shape[-1])
    return pca_proj

# @memory.cache
def get_pcs_stefan(X, pcs, original_shape=True, return_projectors=False):
    """
        Return principal components of X (Stefan's version).
        Args:
            X ([num_samples, ambient space dimension]): Data matrix of samples where each sample corresponds to a row of
                X.
            pcs ([num_pcs,]): List of principal components to return.

        Returns:
            pca_proj: ([num_pcs, ambient space dimension]): Projection of X onto principal components given by pcs.

        """
    # Todo: sort eigenvalues by magnitude
    X_shape = X.shape
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
        print("Warning: concatenated first however many dimensions to get square data array")
    X = X - np.mean(X, axis=0)
    cov = np.cov(X.T)
    eig, ev = np.linalg.eig(cov)
    eig = np.real(eig)
    ind = np.argsort(np.abs(eig))[::-1]
    ev = np.real(ev[:, ind])
    # pca_proj = np.dot(ev[:, pcs].T, X.T)
    pca_proj = np.dot(X, ev[:, pcs])
    if original_shape:
        pca_proj = pca_proj.reshape(*X_shape[:-1], pca_proj.shape[-1])
    if return_projectors:
        return pca_proj, ev
    else:
        return pca_proj

# @memory.cache
def get_right_vecs_stefan(X):
    """
        Return right singular vectors of X (Stefan's version).
        Args:
            X ([num_samples, ambient space dimension]): Data matrix of samples where each sample corresponds to a row of
                X.
            pcs ([num_pcs,]): List of principal components to return.

        Returns:
            pca_proj: ([num_pcs, ambient space dimension]): Projection of X onto principal components given by pcs.

        """
    # Todo: sort eigenvalues by magnitude
    X_shape = X.shape
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
        print("Warning: concatenated first however many dimensions to get square data array")
    X = X - np.mean(X, axis=0)
    cov = np.cov(X.T)
    eig, ev = np.linalg.eig(cov)
    eig = np.real(eig)
    ind = np.argsort(np.abs(eig))[::-1]
    ev = np.real(ev[:, ind])
    return ev


def get_ordered_eigens(X, symmetric_matrix=False):
    """
        Return ordered eigenvalues and eigenvectors of square matrix X.
        Args:
            X ([num_samples, ambient space dimension]): Square matrix to get eigenvectors and eigenvalues.

        Returns:
            eig: eigenvalues
            ev: eigenvectors

        """
    # Todo: sort eigenvalues by magnitude
    if symmetric_matrix:
        eig, ev = np.linalg.eigh(X)
    else:
        eig, ev = np.linalg.eig(X)
    # eig = np.real(eig)
    ind = np.argsort(np.abs(eig))[::-1]
    # ev = np.real(ev[:, ind])
    ev = ev[:, ind]
    eig = eig[ind]
    return eig, ev

# @memory.cache
def get_effdim_and_radius(X):
    X_centered = X - np.mean(X, axis=0)
    U, s, Vt = np.linalg.svd(X_centered)
    D_eff = np.sum(s ** 2) ** 2 / np.sum(s ** 4)
    R_eff = np.sqrt(np.sum(s ** 4) / np.sum(s ** 2))
    return D_eff, R_eff

def get_effdim_and_radius_cov(X):
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X.T)
    eig, ev = np.linalg.eig(cov)
    eig = np.real(eig)
    # ind = np.argsort(np.abs(eig))[::-1]
    # eig = eig[ind]
    D_eff = np.sum(eig) ** 2 / np.sum(eig ** 2)
    R_eff = np.sqrt(np.sum(eig ** 2) / np.sum(eig))
    return D_eff, R_eff

# @memory.cache
def get_effdim(X):
    # X_centered = X - np.mean(X, axis=0)
    # C = X_centered.T @ X_centered / (X.shape[0]-1)
    C = np.cov(X, rowvar=False)
    D_eff = np.trace(C) ** 2 / np.trace(C @ C)
    # if np.trace(C@C) < 10e-6:
    #     print()
    # print(np.trace(C@C))
    # U, s, Vt = np.linalg.svd(X_centered)
    # D_eff = np.sum(s ** 2) ** 2 / np.sum(s ** 4)
    # R_eff = np.sqrt(np.sum(s ** 4) / np.sum(s ** 2))
    return D_eff


# def get_num_clusters(X):
#     D = squareform(pdist(X))


if __name__ == '__main__':
    # get_pcs.clear()
    np.random.seed(1)
    X = np.random.randn(1000, 40)
    Y = get_pcs(X, [0, 1])
    print(Y)
