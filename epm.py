import scipy.io
import subprocess
import warnings
import tempfile
import numpy as np
import scipy.sparse as sp

def epm(A, num_coms, X=None, threshold=1.0, dir='/media/d/uni/kdd/EPM', datatype='binary', modeltype='Infinte', burnin=1500, collections=1500, train_ratio=1.0):
    """
    Performs the EPM model.

    Parameters:
    -----------
    A : sp.sparse_matrix, shape [N, N]
        Symmetric, binary adjacency matrix.
    num_coms : int
        The number of communities in the graph.
    X : ndarray, shape [N, D]
        Features, unused.
    threshold : float
        The minimal latent count for community assignment of nodes.
    dir : string
        Directory of the wrapped script files. See https://github.com/WodkaRHR/EPM
    datatype : string
        The datatype parameter for the EPM model.
    modeltype : string
        The modeltype parameter for the EPM model.
    burnin : int
        Number of burnin iterations.
    collections : int
        Number of collection iterations.
    train_ratio : float
        The number of links to include while iterating.
    
    Returns:
    --------
    z : sp.sparse_matrix, shape [N, k]
        The communitiy 
    """
    return _epm(A, num_coms, X=X, hierarchical=False, threshold=threshold, dir=dir, datatype=datatype, modeltype=modeltype, burnin=burnin, collections=collections, train_ratio=train_ratio)

def hepm(A, num_coms, X=None, threshold=1.0, dir='/media/d/uni/kdd/EPM', datatype='binary', modeltype='Infinte', burnin=1500, collections=1500, train_ratio=1.0):
    """
    Performs the HEPM model.

    Parameters:
    -----------
    A : sp.sparse_matrix, shape [N, N]
        Symmetric, binary adjacency matrix.
    num_coms : int
        The number of communities in the graph.
    X : ndarray, shape [N, D]
        Features, unused.
    threshold : float
        The minimal latent count for community assignment of nodes.
    dir : string
        Directory of the wrapped script files. See https://github.com/WodkaRHR/EPM
    datatype : string
        The datatype parameter for the HEPM model.
    modeltype : string
        The modeltype parameter for the HEPM model.
    burnin : int
        Number of burnin iterations.
    collections : int
        Number of collection iterations.
    train_ratio : float
        The number of links to include while iterating.
    
    Returns:
    --------
    z : sp.sparse_matrix, shape [N, k]
        The communitiy 
    """
    return _epm(A, num_coms, X=X, hierarchical=True, threshold=threshold, dir=dir, datatype=datatype, modeltype=modeltype, burnin=burnin, collections=collections, train_ratio=train_ratio)

def _epm(A, num_coms, X=None, hierarchical=True, threshold=1.0, dir='/media/d/uni/kdd/EPM', datatype='binary', modeltype='Infinte', burnin=1500, collections=1500, train_ratio=1.0):
    # Helper method for more clean code (interface of EPM and HEPM are very similar)
    with tempfile.NamedTemporaryFile(suffix='.mat') as input, tempfile.NamedTemporaryFile(suffix='.mat') as output:
        
        scipy.io.savemat(input.name, {
            'A' : A,
            'K' : num_coms,
            'Datatype' : datatype,
            'Modeltype' : modeltype,
            'Burnin' : burnin,
            'Collections' : collections,
            'TrainRatio' : train_ratio,
        })
        wrapper = 'HGP_EPM_wrap' if hierarchical else 'GP_EPM_wrap'
        matlab_command = f'"{wrapper}(\'{input.name}\', \'{output.name}\');exit;"'
        cmd = ['matlab', '-nodisplay', '-nodesktop', '-r', matlab_command]
        p = subprocess.Popen(' '.join(cmd), cwd=dir, shell=True)
        return_code = p.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, ' '.join(cmd))

        # Read results
        result = 'm_i_k_dot_dot' if hierarchical else 'mi_dot_k'
        return (sp.csr_matrix(scipy.io.loadmat(output.name)[result]).T > threshold).astype(int)


# Test 
if __name__ == '__main__':
    A = np.array(np.random.randn(50, 50) > 1, dtype=int)
    num_coms = 3
    Z = epm(A, num_coms)
    print(Z.shape, Z.nnz)
    Z = hepm(A, num_coms)
    print(Z.shape, Z.nnz)
