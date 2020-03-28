import sys
import os
# import autograd.numpy as  # Need this for autograd fanciness.
import numpy as np
import autograd
from joblib import Memory
from mython import check_convergence as check_convergence

cache_dir = '../joblib_cache'
os.makedirs(cache_dir, exist_ok=True)

memory = Memory(cachedir=cache_dir, verbose=1)

# %% Funcions for running the dynamics of the network

def phi(x):
    """nonlinearity"""
    return np.tanh(x)

def phiPrime(x):
    """derivative of nonlinearity"""
    return 1. / np.cosh(x) ** 2

# def oneStep(h, W, b, phi=phi):  # Todo: work out W vs W.T
#     """one step evolution of RNN (without input or output). Assumes dt=1."""
#     return phi(W.dot(h) + b)

def oneStep(h, W, b, x, Win=0, phi=phi):  # Todo: work out W vs W.T
    """one step evolution of RNN (without input or output). Assumes dt=1."""
    # return phi(W.dot(h) + np.dot(Win, x) + b)
    return phi(h @ W.T + x @ Win.T + b)

# def oneStepVar(x, Y, W, b, phiPrime=phiPrime):  # Todo: work out W vs W.T
#     """one step evolution of associated variational equation (without input or output). Assumes dt=1"""
#     # v = phiPrime(W.dot(x) + b)
#     # A = np.multiply(v, W.T).T
#     v = phiPrime(W.dot(x) + Win.dot(x) + b)
#     A = np.multiply(v, W.T).T
#     return A.dot(Y)

def oneStepVar(h, Y, W, b, Win=0, x=0, phiPrime=phiPrime):  # Todo: work out W vs W.T
    """one step evolution of associated variational equation (without input or output). Assumes dt=1"""
    # v = phiPrime(W.dot(x) + b)
    # A = np.multiply(v, W.T).T
    v = phiPrime(W.dot(h) + np.dot(Win, x) + b)
    A = np.multiply(v, W.T).T
    return A.dot(Y)

def net_f(h, Wrec, Win, b, x, t, phi=phi):
    k = int(np.floor(t))
    x_t = x[k]
    return -h + phi(np.dot(h, Wrec) + np.dot(x_t, Win) + b)

def net_f_auton(h, Wrec, b, Win=0, x=0, phi=phi):  # Autonomous, constant input
    return -h + phi(np.dot(h, Wrec) + np.dot(x, Win) + b)

def net_f_full(h, Wrec, Win, b, X, t, phi=phi):
    k = int(t)
    X_t = X[:, k]
    return -h + phi(np.dot(h, Wrec) + np.dot(X_t, Win) + b)

def net_out_full(h, Wrec, Win, Wout, b, X, t, phi=phi):
    k = int(t)
    X_t = X[:, k]
    hid = -h + phi(np.dot(h, Wrec) + np.dot(X_t, Win) + b)
    out = np.dot(hid, Wout)
    return out

def net_f_full_auton(h, Wrec, b, Win=0, X=0, phi=phi):
    return -h + phi(np.dot(h, Wrec) + np.dot(X, Win) + b)

def net_f_auton_jacob(h, Wrec, b, Win=0, x=0, phiPrime=phiPrime):
    N = h.shape[0]
    v = phiPrime(np.dot(h, Wrec) + np.dot(x, Win) + b)
    A = -np.eye(N) + np.multiply(v, Wrec).T
    return A

# I = np.eye(N)
#
#
# def gp(x):
#     return np.cos(x) ** (-2)
#
#
# def jac_f(h):
#     return -I + np.dot(gp(np.dot(h, Wrec) + b), Wrec.T)


# %% Functions needed for LE compute


# def oneStepVarQR(x, Q, W, b, phiPrime=phiPrime):
#     """one step variational equation with QR decomposition"""
#     v = phiPrime(W.dot(x) + b)
#     A = np.multiply(v, W.T).T  # This is equivalent to (np.diag(v) @ W.T).T
#     Z = A.dot(Q)
#     q, r = np.linalg.qr(Z, mode='complete')
#     q = q[:, :k_LE]
#     s = np.diag(np.sign(np.diag(r)))
#     return q.dot(s), np.diag(r.dot(s))

def oneStepVarQR(h, Q, W, b, Win=0, x=0, phiPrime=phiPrime, k_LE=None):
    """one step variational equation with QR decomposition"""
    if k_LE is None:
        k_LE = len(h)
    # v = phiPrime(W.dot(x) + b)
    # A = np.multiply(v, W.T).T  # This is equivalent to (np.diag(v) @ W.T).T
    # Z = A.dot(Q)
    Z = oneStepVar(h, Q, W, b, Win, x, phiPrime=phiPrime)
    q, r = np.linalg.qr(Z, mode='complete')
    q = q[:, :k_LE]
    return q, np.abs(np.diag(r))

@memory.cache()
def getSpectrum(W, b, Win=0, x=0, k_LE=None, max_iters=1000, tol=1e-3, max_ICs=10, ICs=None, verbose=False):
    """
    Iteration of variational equation and estimation of LE spectrum (first k_LE exps).

    Args:
        W (array): Recurrent weights connecting hidden units.
        b (array): Bias.
        Win (array): Input weights
        x (array): Input data
        k_LE (int or None): Number of LEs to compute.
        max_iters (int): Maximum number of iterations (in time).
        tol (float): Tolerance for standard error of the mean between initial conditions before stopping.
        max_ICs (int): Maximum number of initial conditions to try in computing standard error of the mean.
        ICs (array or none): The initial conditions for the hidden units to use. If None they are chosen randomly.
        verbose (bool):

    Returns:
        (array): Estimate Lyapunov exponent spectrum averaged across initial conditions.
        sem (float): Standard error of the mean across initial conditions.
        traj (array): Trajectories that the network follows in calculating the Lyapunov exponents.
    """

    if k_LE is None: k_LE = W.shape[0]

    # loop over multiple initial conditions
    specs = []  # container for spectra accross multiple ICs
    sem = 100  # average standard error of the mean (initialize at large dummy value)
    sems = []
    IC_cnt = 0

    traj = np.zeros((max_ICs, max_iters + 1, W.shape[0]))
    # if return_traj is False:
    #     traj = None

    if ICs is None:
        ICs = np.random.uniform(-.2, .2, (max_iters, W.shape[0]))

    if not hasattr(x, '__len__'):
        x = x * np.ones(Win.shape[1])

    iter_cc = check_convergence.CheckConvergence(memory_length=10, tolerance=.1*tol)
    IC_cc = check_convergence.CheckConvergence(memory_length=None, tolerance=tol, min_length=4)
    sem_converge_bool = False

    while IC_cnt < max_ICs and not sem_converge_bool:
        it = 0  # time iteration counter
        h = ICs[IC_cnt]
        traj[IC_cnt, 0] = h
        Q = np.eye(W.shape[0])
        Q = Q[:, :k_LE]  # Initialize othonormal matrix
        running_log_sum = np.zeros(k_LE)
        iter_cc.clear_memory()
        converge_bool = False
        while it < max_iters and not converge_bool:
            it += 1
            h = oneStep(h, W, b, x, Win)
            traj[IC_cnt, it] = h
            Q, r = oneStepVarQR(h, Q, W, b, Win, x, k_LE=k_LE)
            running_log_sum += np.log2(r)
            converge_bool = iter_cc.check_convergence(running_log_sum / it, comparison_mode='sem')
        LEs = running_log_sum / it
        specs.append(LEs)  # save spectrum
        IC_cnt += 1  # initial condition counter
        sem_converge_bool, sem = IC_cc.check_convergence(LEs, comparison_mode='sem', ret_dev=True)
        # if IC_cnt > 1: sem = np.mean(np.std(np.vstack(specs), axis=0)) / np.sqrt(IC_cnt)
        sems.append(sem)
        if verbose: print('%d/%d, SEM:%.5f' % (IC_cnt, max_ICs, sem))

    print('standard error: %.5f' % sem)

    return np.mean(specs, axis=0), sem, traj

if __name__ == '__main__':
    # %% Test out the jacobians

    def gen_net_f_auton_jac(h, Wrec, b, Win=0, x=0, phi=phi):
        """
        generate net_f_auton function's Jacobian.

        Args:
            h ():
            Wrec ():
            b ():
            Win ():
            x ():
            phi ():

        Returns:
            net_f_auton_jac (function): Jacobian for net_f_auton.

        """

        def net_f_auton_reduced(h):
            return net_f_auton(h, Wrec, b, Win, x, phi)

        f_auton_jac = autograd.jacobian(net_f_auton_reduced)
        return f_auton_jac

    N = 4
    Nx = 3
    Win = np.random.randn(N, Nx).T
    Wrec = np.random.randn(N, N)
    h = np.random.randn(N)
    x = np.random.randn(Nx)
    b = 0

    d1 = net_f_auton_jacob(h, Wrec, b, Win=Win, x=x, phiPrime=phiPrime)
    temp = gen_net_f_auton_jac(h, Wrec, b, Win=Win, x=x, phi=phi)
    d2 = temp(h)
    print(d1)
    print(d2)
