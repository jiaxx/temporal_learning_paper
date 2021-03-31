import functools
import copy
import numpy as np
import itertools
import scipy.stats as stats
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
try:
    from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
except ImportError:
    from sklearn.metrics.pairwise import pairwise_distance_functions\
            as PAIRWISE_DISTANCE_FUNCTIONS
from joblib import Parallel, delayed

from yamutils.fast import reorder_to

# (NOTE) Use this version: https://github.com/yamins81/scikit-data
import skdata.larray as larray
from skdata.data_home import get_data_home
from skdata.utils.download_and_extract import download_boto, extract

from . import classifier
from . import kanalysis

try:
    import bangmetric as bm
except ImportError:
    print("Can't import bangmetric")

DEF_BITER = 900    # default bootstrapping iters

"""
Confusion Matrix Format:
shape = [n_class (pred), n_class (true)]
-Rows: Predicted/Chosen label
-Columns: True label
"""

def bstrap_metric(CM, fn, fn_kw, biter=DEF_BITER, seed=0):
    """
    Bootstrapping interface to get population varability
    of CMs (aka. type 1 error)
    """
    if CM.ndim == 2:
        CM = CM[:, :, np.newaxis]
        biter = 2   # should be > 1 due to ddof=1 in np.std
    assert CM.ndim == 3

    rng = np.random.RandomState(seed)

    res = []
    n = CM.shape[-1]
    for _ in xrange(biter):
        ii = rng.randint(n, size=n)
        res.append(fn(CM[:, :, ii], **fn_kw))

    return np.array(res)


def bstrap_consistency(datX, datY, biter=DEF_BITER, seed=0):
    """
    Bootstrapping interface to compute the "standard" task
    varability (of consistency) (aka. type 2 error)
    """
    datX = np.array(datX).ravel()
    datY = np.array(datY).ravel()
    assert len(datX) == len(datY)
    rng = np.random.RandomState(seed)

    res = []
    n = len(datX)
    for _ in xrange(biter):
        ii = rng.randint(n, size=n)
        res.append(consistency(datX[ii], datY[ii]))

    return np.array(res)


def composite_self_consistency(CMs, m0, fn, fn_kw, biter=DEF_BITER, seed=0):
    """
    Composite bootstrapping-like interface to get the self consistency of
    a "typical resampled composite individual" in CMs against m0

    biter parameter controls the number of "fake" individuals.
    """
    assert all([CM.ndim == 3 for CM in CMs])
    rng = np.random.RandomState(seed)

    res = []

    #loop over "fake" individuals
    for _ in xrange(biter):
        m = []

        # get metric values m of a "composite individual" by randomly
        # sampling which individual to take from independently for each task
        for CM in CMs:
            n = CM.shape[-1]
            i = rng.randint(n)
            m.extend(fn(CM[:, :, i], **fn_kw))

        res.append(consistency(m, m0))

    return np.array(res)


def dprime(X, population_avg=True, use_zs=True, maxv=None,
        minv=None, cor=False):
    """
    CM.shape = (num_classes, num_classes, population_size)
    """

    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    assert X.ndim == 3, X.shape

    if population_avg == 'pop':
        return dprime(X.sum(2)[:, :, np.newaxis], population_avg=True,
                      use_zs=use_zs, maxv=maxv, minv=minv, cor=cor)
    if maxv is None:
        maxv = np.inf
    if minv is None:
        minv = -np.inf

    H = X.diagonal(axis1=0, axis2=1).T / X.sum(0).astype(float)
    FA = (X.sum(1) - X.diagonal(axis1=0, axis2=1).T) / \
            (X.sum(0).sum(0) - X.sum(0)).astype(float)
    FA = np.where(np.isnan(FA) | np.isinf(FA), 0, FA)
    if cor:
        N0 = float(X.shape[0])
        FA = np.minimum(FA * (N0 / (N0 - 1)), 1)

    if use_zs:
        zs = stats.norm.ppf
    else:
        zs = lambda x: x

    if population_avg == 'after':
        return zs(np.maximum((H - FA).mean(1), 0)).clip(minv, maxv)
    elif population_avg:
        Hs = zs(H.mean(1)).clip(minv, maxv)
        FAs = (-1*zs(FA.mean(1))).clip(minv, maxv)
        return Hs + FAs
    else:
        Hs = zs(H).clip(minv, maxv)
        FAs = (-1*zs(FA)).clip(minv, maxv)
        return (Hs + FAs).mean(1)


def symmetrize_confusion_matrix(CM, take='all'):
    """
    Sums over population, symmetrizes, then return upper triangular portion
    :param CM: numpy.ndarray confusion matrix in standard format
    """
    if CM.ndim > 2:
        CM = CM.sum(2)
    assert len(CM.shape) == 2, 'This function is meant for single subject confusion matrices'
    symmetrized = CM+CM.T
    #print symmetrized
    #print np.triu_indices(CM.shape[0])
    if take == 'all':
        rval = symmetrized[np.triu_indices(CM.shape[0])]
    elif take == 'diagonal':
        rval = symmetrized.diagonal()
    elif take == 'off_diagonal':
        rval = symmetrized[np.triu_indices(CM.shape[0], 1)]
    else:
        raise ValueError("Take %s not recognized. Allowed takes are all, diagonal and off_diagonal" % take)
    return rval


def symmetrized_dprime(X):
    X = X.sum(2)
    Xw = X / X.sum(0).astype(float)
    #Z = Xw + Xw.swapaxes(0, 1)
    Z = Xw + Xw.T
    #d = dprime(Z, use_zs=False, population_avg='mean')
    d = dprime(Z, use_zs=False, population_avg='pop')
    d[np.isnan(d)] = 0
    return d


def performances(X):
    """
    CM.shape = (num_classes, num_classes, population_size)
    """
    X = X.sum(2)
    E = X.sum(1) + X.sum(0) - 2 * X.diagonal()
    T = X.sum().astype(np.float)
    P = (T - E) / T
    return P


def dprime_bangmetric(X, population_avg='mean_rate', use_zs=True,\
        maxv=np.inf, minv=-np.inf, cor=False, **kwargs):
    """New implementation of dprime() using "bangmetric" --- designed
    as interchangable as possible with the original dprime().

    Note that major differences will be arising from that this function
    will use fudge factors by default.  To disable fudge factors, set
    `fudge_mode='none'` in `kwargs`.

    If `collation` is not given in `kwargs`, this function will compute
    one-vs-rest d' for each class as the original dprime() did.

    Parameters
    ----------
    X: array-like, shape = [n_class (pred), n_class (true), n_population] or
        shape = [n_class (pred), n_class (true)]
        Confusion matrix

    population_avg: string, optional (default='mean_rate')
        Can be one of the followings:
        * 'mean_rate' (or True, for backward compatibility): compute the
          mean TPR and FPR across subjects
        * 'mean_dp' (or False): compute the mean d' across subjects
        * 'sum_subj' (or 'pop'): take the sum across subjects to get
          the "summed" confusion matrix used for d' calculation
        * 'after': clip the final value after averaging across subjects
          (ask Dan for details!)

    use_zs: bool, optional (default=True)
        If True, use z-function to compute d'.

    maxv: float, optional (default=np.inf)
        Maximum possible rate

    minv: float, optional (defaut=-np.inf)
        Minimum possible rate

    cor: bool, optional (default=False)
        A heuristic to account the sizes of classes for better optimization.
        Ask Dan for details.

    kwargs: named arguments, optional
        Passed to ``bangmetric.confusion_matrix_stats()``.
        By assigning ``collation``, ``fudge_mode``, and ``fudge_factor``
        one can change the behavior of d-prime computation
        (see ``bangmetric.confusion_matrix_stats()`` for details).

    Returns
    -------
    dp: array, shape = [n_groupings]
        An array of d-primes, where each element
        corresponds to each grouping of positives and negatives.
        (by default, each elem means each class.)
    """

    X = np.array(X)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    assert X.ndim == 3, X.ndim

    if population_avg in ['sum_subj', 'pop']:    # population summation averaging
        X = X.sum(2)[:, :, np.newaxis]
        population_avg = 'mean_rate'

    # bangmetric.confusion_matrix_stats expects (true, pred) confu matrix!!
    XT = X.T
    # R: shape = (n_population, 6 (meaning P, N, TP, TN, FP, FN), n_grouping)
    # ...and n_grouping is determined by `collation` (see Doc), but by default
    # n_grouping = n_class (as this function will assume one-vs-rest)
    R  = np.array([bm.confusion_matrix_stats(XT0, **kwargs) for XT0 in XT])
    # Each of P, N, ... will have the shape = (n_grouping, n_population)
    P, N, TP, _, FP, _ = np.rollaxis(R.T, 1)

    TPR = TP / P
    FPR = FP / N
    if cor:
        n_class = float(X.shape[0])
        FPR = np.minimum(FPR * (n_class / (n_class - 1)), 1)

    if use_zs:
        dp = lambda TPR, FPR: bm.dprime(TPR, FPR, mode='rate',\
                max_ppf_value=maxv, min_ppf_value=minv)
    else:
        dp = lambda TPR, FPR: np.clip(TPR, minv, maxv) - \
                np.clip(FPR, minv, maxv)

    if population_avg == 'after':
        qTPR = np.maximum((TPR - FPR).mean(1), 0)  # quasi-TPR
        qFPR = np.zeros(qTPR.shape)
        return dp(qTPR, qFPR)

    elif population_avg in ['mean_rate', True]:
        TPR = TPR.mean(1)
        FPR = FPR.mean(1)
        return dp(TPR, FPR)

    elif population_avg in ['mean_dp', False]:
        return dp(TPR, FPR).mean(1)

    else:
        raise ValueError('Invalid mode')


def performances_bangmetric(X, **kwargs):
    """New implementation of performances() using "bangmetric."

    If `collation` is not given in `kwargs`, this function will compute
    one-vs-rest accuracies for each class as the original performances() did.

    Parameters
    ----------
    X: array-like, shape = [n_class (pred), n_class (true), n_population] or
        shape = [n_class (pred), n_class (true)]
        Confusion matrix

    kwargs: named arguments, optional
        Passed to ``bangmetric.accuracy()``.
        By assigning ``collation`` and ``balanced``
        one can change the behavior of accuracy computation.
        Especially:

            balanced: bool, optional (default=False):
                Computes the balanced accuracy.

        See ``bangmetric.accuracy()`` for details.

    Returns
    -------
    acc: array, shape = [n_groupings]
        An array of accuracies, where each element corresponds to each
        grouping of positives and negatives.
        (by default, each elem means each class.)
    """

    X = np.array(X)
    if X.ndim == 3:
        X = X.sum(2)
    assert X.ndim == 2
    XT = X.T  # necessary b/c bangmetric convention is (true, pred)
    acc = bm.accuracy(XT, mode='confusionmat', **kwargs)
    return acc


def consistency(A, B):
    """Compute the consistency between A and B using "bangmetric."

    Parameters
    ----------
    A: array-like, shape = [n_task]
        An array of task performances (e.g., d's)

    B: array-like, shape = [n_task],
        An array of task performances
        The order of tasks should be the same as A.

    Returns
    -------
    consistency: float
        The consistency between A, B

    """
    return bm.consistency(A, B)


def consistency_from_confusion_mat(CMs1, CMs2, metric='dp_standard',
                                   metric_kwargs=None,
                                   error='std',
                                   error_kwargs=None,
                                   error_finite=True,
                                   verbose=False):
    """Compute the consistency from two sets of consution matrices

    Parameters
    ----------
    CMs1: list, length = n_confusion_matrices
        Confusion matrices for one "population" on a list of tasks. That is:

           CMs1 = [cm11, cm12, ... cm1n]

        where n = number of tasks, and each cm1i is a 3-dimensional confusion
        matrix of the form

           cm1i.shape = (num_classes_i, num_classes_i, pop_size_i)

        where num_classes_i = number of classes in the ith task and
        pop_size_i = number of "individuals" in the population that was
        tested on the ith task.

        Generally, "population" means subjects in the human case
        or unit groupings/image splits in the feature case.

    CMs2: list, length = n_confusion_matrices
        A list of confusion matrices for a second population, on the same tasks
        as CMs1.  The first two dimensions of each CMs2 matrix should have the
        same shape as corresponding matrices in CMs1, because they're supposed
        to represent to the same task (and therefore have the same number of
        classes).

        For example, if the (first two dimension of) CMs1 is
            [8x8 matrix, 5x5 matrix, 18x18 matrix],
        then CMs2 must be
            [a 8x8 matrix, a 5x5 matrix, a 18x18 matrix]
        ... although the third dimension of each matrix (the number of
        "individuals" in each population) can be different between CMs1 and Cms2.

    metric: string, optional (default='dp_standard')
        Which metric to compute from confusion matrices.
        Options:
            'dp_standard': the "standard" d's used in the HvM paper
            'dp': d's
            'acc': accuracies (a.k.a. performances)
            'raw': element-wise comparison
            'symmetrized_raw': element-wise comparison of symmetrized confusion matrix

    metric_kwargs: named arguments, optional
        Passed to the metric functions (e.g., dprime_bangmetric() or
        performances_bangmetric()).  Useful when ``metric`` is 'dp' or
        'acc' to finely control the behavior of metric functions.

    error: string, optional (default='std')
        whether to use "mean/std"-based statistics or "median-difference"
        based statistics.

    error_kwargs: dict, optional (default={})
        keyword arguments passed to the various boostrapping functions
        to control boostrapping parameters in computation of errors


    Returns
    -------
    c: float
        The consistency between (population aggregate) values of
        metric vectors computed from CMs1 and CMs2.

    m1: list, length = n_tasks
        A vector of metric values for each task given by CMs1, taken
        at the population-aggregate level.  Note that this may not be the same
        as averaging the metric value over the population since the aggregation
        may occur at the confusion-matric level prior to computing the metric.

    m2: list, length = n_tasks
        A list of metric values for tasks given by CMs2

    sc1: float
        The self-consistency of CMs1.
        That is, create a bunch of "fake individuals" with values for each
        task, drawn as composites of individuals the CM1 population, and
        compute for each such fake individual the consistency of its metric
        vector with the population aggregate metric vector m1.  A central
        tendency is then computed for this over all fake individuals.  (Note
        that if there were lots of "real" individuals that were evaluated on
        ALL tasks, e.g. if the pop_size_i was the same for each i with
        corresponding individuals in each position in the third dimensions of
        the CMs, then it might be better simply to compute with these
        individuals instead of creating fake individuals by composite.)

    sc2: float
        The self-consistency of CMs2

    ce: float
        The task-induced error of the consistency between CM1 and CM2
        population

    m1e: list, length = n_tasks
        A list of errors due to population choice in the estimates of
        population-level aggregates m1.

    m2e: list, length = n_tasks
        A list of errors for m2 based on population choice

    sc1e: float
        The error of sc1.

    sc2e: float
        The error of sc2.
    """

    if metric_kwargs is None:
        metric_kwargs = {}
    if error_kwargs is None:
        error_kwargs = {}

    for M1, M2 in zip(CMs1, CMs2):
        assert M1.shape[:2] == M2.shape[:2]

    if verbose:
        print('Computing cmatrix1 metrics ... ')
    m1, sc1, m1e, sc1e = metrics_from_confusion_mat(CMs1, metric=metric,
                                   metric_kwargs=metric_kwargs,
                                   error=error,
                                   error_kwargs=error_kwargs,
                                   error_finite=error_finite,
                                   verbose=verbose)
    if verbose:
        print('Computing cmatrix2 metrics ... ')
    m2, sc2, m2e, sc2e = metrics_from_confusion_mat(CMs2, metric=metric,
                                   metric_kwargs=metric_kwargs,
                                   error=error,
                                   error_kwargs=error_kwargs,
                                   error_finite=error_finite,
                                   verbose=verbose)

    c = consistency(m1, m2)
    ce = None
    if error is not None and error != 'none':
        # -- init
        if error == 'std':
            errfn = lambda x: np.std(x, ddof=1)
        elif error == 'mad':
            errfn = lambda x: np.median(np.abs(x - np.median(x)))
        else:
            raise ValueError('Not recognized "error"')

        def collect_err(x):
            if error_finite:
                x = x[np.isfinite(x)]
            return errfn(x)

        #task-induced error for consistency between m1 and m2
        if verbose:
            print('Computing task consistency ...')
        c_bs = bstrap_consistency(m1, m2, **error_kwargs)
        if verbose:
            print('... done.')
        # -- finally get actual error values
        ce = collect_err(c_bs)

    return c, m1, m2, sc1, sc2, ce, m1e, m2e, sc1e, sc2e


def metrics_from_confusion_mat(CMs, metric='dp_standard',
                                   metric_kwargs=None,
                                   error='std',
                                   error_kwargs=None,
                                   error_finite=True,
                                   verbose=False):
    """Compute the consistency from two sets of consution matrices

    Parameters
    ----------
    CMs: list, length = n_confusion_matrices
        Confusion matrices for one "population" on a list of tasks. That is:

           CMs = [cm1, cm12, ... cmn]

        where n = number of tasks, and each cmi is a 3-dimensional confusion
        matrix of the form

           cmi.shape = (num_classes_i, num_classes_i, pop_size_i)

        where num_classes_i = number of classes in the ith task and
        pop_size_i = number of "individuals" in the population that was
        tested on the ith task.

        Generally, "population" means subjects in the human case
        or unit groupings/image splits in the feature case.

    metric: string, optional (default='dp_standard')
        Which metric to compute from confusion matrices.
        Options:
            'dp_standard': the "standard" d's used in the HvM paper
            'dp': d's
            'acc': accuracies (a.k.a. performances)
            'raw': element-wise comparison
            'symmetrized_raw': element-wise comparison of symmetrized confusion matrix

    metric_kwargs: named arguments, optional
        Passed to the metric functions (e.g., dprime_bangmetric() or
        performances_bangmetric()).  Useful when ``metric`` is 'dp' or
        'acc' to finely control the behavior of metric functions.

    error: string, optional (default='std')
        whether to use "mean/std"-based statistics or "median-difference"
        based statistics.

    error_kwargs: dict, optional (default={})
        keyword arguments passed to the various boostrapping functions
        to control boostrapping parameters in computation of errors


    Returns
    -------

    m: list, length = n_tasks
        A vector of metric values for each task given by CMs, taken
        at the population-aggregate level.  Note that this may not be the same
        as averaging the metric value over the population since the aggregation
        may occur at the confusion-matric level prior to computing the metric.

    sc: float
        The self-consistency of CMs.
        That is, create a bunch of "fake individuals" with values for each
        task, drawn as composites of individuals the CM population, and
        compute for each such fake individual the consistency of its metric
        vector with the population aggregate metric vector m1.  A central
        tendency is then computed for this over all fake individuals.  (Note
        that if there were lots of "real" individuals that were evaluated on
        ALL tasks, e.g. if the pop_size_i was the same for each i with
        corresponding individuals in each position in the third dimensions of
        the CMs, then it might be better simply to compute with these
        individuals instead of creating fake individuals by composite.)

    me: list, length = n_tasks
        A list of errors due to population choice in the estimates of
        population-level aggregates m.

    sce: float
        The error of sc.

    """

    if metric_kwargs is None:
        metric_kwargs = {}
    if error_kwargs is None:
        error_kwargs = {}

    m = []   # metrics for CMs

    kwargs = copy.deepcopy(metric_kwargs)
    if metric == 'dp_standard':
        metric_func = dprime_bangmetric
        # "standard" options
        kw = {}
        kw['population_avg'] = 'sum_subj'
        kw['fudge_mode'] = 'correction'
        kw['fudge_factor'] = 0.5
        kwargs.update(kw)
    elif metric == 'dp':
        metric_func = dprime_bangmetric
    elif metric == 'acc':
        metric_func = performances_bangmetric
    elif metric == 'raw':
        metric_func = np.ravel
        kwargs = {}   # discard all kwargs
    elif metric == 'symmetrized_raw':
        metric_func = symmetrize_confusion_matrix
        kwargs = {'take': 'all'}
    elif metric == 'diagonal':
        metric_func = symmetrize_confusion_matrix
        kwargs = {'take': 'diagonal'}
    elif metric == 'off_diagonal':
        metric_func = symmetrize_confusion_matrix
        kwargs = {'take': 'off_diagonal'}

    else:
        raise ValueError('Not recognized "metric"')

    for M in CMs:
        m.extend(metric_func(M, **kwargs))

    me, sc, sce = None, None, None

    if error is not None and error != 'none':
        # -- init
        if error == 'std':
            errfn = lambda x: np.std(x, ddof=1)
            ctrfn = lambda x: np.mean(x)
        elif error == 'mad':
            errfn = lambda x: np.median(np.abs(x - np.median(x)))
            ctrfn = lambda x: np.median(x)
        else:
            raise ValueError('Not recognized "error"')

        bstrap_metric_helper = lambda x: bstrap_metric(x, metric_func,
                kwargs, **error_kwargs).T

        def collect_err(x):
            if error_finite:
                x = x[np.isfinite(x)]
            return errfn(x)

        def collect_ctr(x):
            if error_finite:
                x = x[np.isfinite(x)]
            return ctrfn(x)

        # -- get bootstrapped quantities
        #  the shape of m1e_bs and m2e_bs is [# CMs, # tasks in each CM, # biter]
        if verbose:
            print('Computing bootstrapped metrics ... ')
        me_bs = map(bstrap_metric_helper, CMs)
        me_bs = np.concatenate(me_bs)

        if verbose:
            print('Computing self consistency ...')
        sc_bs = composite_self_consistency(CMs, m, metric_func, kwargs, **error_kwargs)
        if verbose:
            print('... done.')

        # -- finally get actual error values
        me = np.array(map(collect_err, me_bs))
        sc = collect_ctr(sc_bs)
        sce = collect_err(sc_bs)

    return m, sc, me, sce


def get_collation(meta, catfunc, subset_q=None):
    """Returns a list of indices for different classes defined by ``catfunc``.

    Parameters
    ----------
    catfunc: (lambda) function
        Defines a class name for a given meta field (the name is somewhat
        misleading --- should be ``clsfunc``, kept due to the
        historical reason)

    subset_q: (lambda) function, optional
        Determines whether a meta field should be included in the result

    Returns
    -------
    indices: list of list
        Contains lists of indices for different classes.  That is:
        [[ind 1 for cls 1, ind 2 for cls 1, ...], ...
         [ind 1 for cls n, ind 2 for cls n, ...]]

    labels: list
        Gives the class name for each element in ``indices``
    """
    if subset_q is None:
        sub = np.arange(len(meta))
    else:
        sub = np.array(map(subset_q, meta)).astype(np.bool)
        sub = np.nonzero(sub)[0]

    cnames = np.array(map(catfunc, meta[sub]))

    labels = np.unique(cnames)
    indices = [sub[cnames == l] for l in labels]

    return indices, labels


def PCA(M, mode='svd'):
    """Performs PCA on the n-by-p data matrix M.
    Rows of M correspond to observations, columns to variables.

    Returns
    -------
     V:
       A p-by-p matrix, each column containing coefficients
       for one principal component.

     P:
       The principal component projections; that is, the
       representation of M in the principal component space.
       Rows correspond to observations, columns to components.

     s:
       A vector containing the eigenvalues
       of the covariance matrix of M.
    """
    M = M - M.mean(axis=0)

    if mode == 'svd':
        V, s, _ = np.linalg.svd(M.T, full_matrices=True)

    elif mode == 'eig':
        s, V = np.linalg.eigh(np.cov(M.T))
        # we need the following to make it comparable to `s` of SVD
        # (but we don't do that)
        # s[s > 0] = np.sqrt(s[s > 0])

    si = np.argsort(-s)
    V = V[:, si]
    s = s[si]
    P = np.dot(M, V) # projection of the data in the PC space

    return V, P, s


def NKlike(F, meta, collation='obj', metric='correlation', summary='pca', top=2,
        top_neu=0, subset_q=None, squareform=False):
    """Computes NK-like (dis)similarity matrix.

    Parameters
    ----------
    collation: string or (lambda) function, optional
      * 'obj': Collate first by objects (default)
      * 'category': Collate first by categories
      * 'img': No collation --- image level computation
      * (lambda) function: passed to get_collation() as ``catfunc``

    metric: string, optional
      * 'correlation': 1 - pearson r (default)
      * 'cov': covariance
      Also all valid `metric` in
      scipy.spatial.distance.pdist is supported.

    summary: string, optional
      * 'pca': use PCA to get the summary statistics of each collation
      * 'pcastddvnorm': PCA + normalization with sqare roots of eigenvalues
        of cov (stddv)
      * 'pcavarnorm': PCA + normalization with eigenvalues of cov (variance)
      * 'mean': take mean

    top: int, optional
      The number of top "eigen-images" to select
      if `summary` is 'pca' or 'pca*norm'.

    top_neu: int, optional
      The number of top "eigen-neurons" to select
      if `summary` is 'pca' or 'pca*norm'.
      By default, do not compute "eigen-neurons."

    subset_q:
      Passed to get_collation()

    squareform:
      If False (default), a condensed pairwise (dis)similiarity vector
      will be returned.

    Returns
    -------
    D: (dis)similarity vector (default, see `sqaureform`) or matrix
    """
    # -- compute "collated" features Fc
    if collation == 'obj':
        collation = lambda x: x['obj']
    elif collation == 'category':
        collation = lambda x: x['category']
    elif collation == 'img':
        pass
    elif callable(collation):
        pass
    else:
        raise ValueError('Invalid collation')

    if collation == 'img':
        if subset_q is not None:
            subinds = np.array(map(subset_q, meta)).astype(np.bool).nonzero()[0]
        else:
            subinds = np.arange(len(F))
        Fc = F[subinds]
        labels = meta['filename'][subinds]
    else:
        inds, labels = get_collation(meta, collation, subset_q)
        Fc = [F[ind] for ind in inds]

    # -- get summary stats
    # --    (if we are performing img-level computation (no collation) there is nothing
    # --    to compute summary statistics over, so we skip this)
    if collation != 'img':
        if summary == 'mean':
            try:
                Fc = np.mean(Fc, axis=1)
            except:
                Fc = np.array([_fc.mean(0) for _fc in Fc])

        elif summary in ['pca', 'pcastddvnorm', 'pcavarnorm']:
            PCs = []

            def comp_PCA(F0, topn):
                # get eigen-neurons
                _, P, s = PCA(F0)
                PC = P[:, :topn]
                s = s[:topn]

                if summary == 'pcastddvnorm':
                    PC = PC * s
                elif summary == 'pcavarnorm':
                    PC = PC * (s ** 2.)
                return PC

            for F0 in Fc:
                if top_neu > 2:
                    F0 = comp_PCA(F0, top_neu)
                    # assert F0.shape[1] == top_neu

                if top > 0:
                    F0 = comp_PCA(F0.T, top).T
                    # assert F0.shape[0] == top

                PCs.append(F0)
            Fc = np.concatenate(PCs, axis=0)
        else:
            raise ValueError('Invalid summary')
    Fcs = Fc.shape
    if len(Fcs) > 2:
        Fc = Fc.reshape((Fcs[0], np.prod(Fcs[1:])))
    assert len(Fc.shape) == 2, Fc.shape

    # -- get distances
    if metric == 'cov':
        # gives square-form
        D = np.cov(Fc)
        if not squareform:
            # force ignore diagonals
            D = distance.squareform(D, checks=False)

    else:
        ### XXX: drop support for pearsonr and keep cov.
        ### if metric == 'pearsonr':
        ###     D = distance.pdist(Fc, metric='correlation')
        ### else:
        ###     D = distance.pdist(Fc, metric=metric)

        # gives condensed form
        D = distance.pdist(Fc, metric=metric)
        if squareform:
            D = distance.squareform(D)

        ### if metric == 'pearsonr':
        ###     D = 1. - D

    return D, labels


def noise_estimation(N, metric='pearsonr', mode='spearman_brown_split_half', Nctr=None,
        center=np.mean, summary_center=np.mean, summary_spread=np.std,
        sync=True, n_jobs=1, n_iter=DEF_BITER):
    """Estimate the self-self consistencies (e.g., correlation coeffs) of
    individual neurons across trials for the given images --- i.e., internal
    consistency.

    Paramters
    ---------
    N: list of array-like
        List of neuronal data.  Each item should be an array and should
        have the shape of (# of trials, # of images, # of neurons).
        Each item represents a subset of neuronal data for different
        images (e.g., V0, V3, V6).  While the # of trials can be different,
        the neurons must be the same across all items in ``N``.
    metric: string, or callable, default='pearsonr'
        Which metric to use to compute the "internal consistency."
        Supported:
            * 'pearsonr' and 'spearmanr'
            * All valid `metric` in sklearn's pairwise_distances()
            * callable, which takes two vectors and gives a distance between
              the two.
    mode: string, default='spearman_brown_split_half'
        Which method to use to compute the "internal consistency."
        Supported:
            * 'bootstrap': This reconstructs two sets of neuronal data by
                bootstrapping over trials and compares those two replicas.
            * 'bootstrap_assume_true_ctr': reconstructs neuronal data
                by bootstrapping over trials and compares with the fixed,
                estimated true central tendency (``Nctr``, see below).
                This assumes we know the true central tendency (which is
                usually NOT TRUE) and therefore always gives the highest
                internal consistency values among all supported modes here.
            * 'spearman_brown_split_half': splits the data into halves,
                computes the consistency between **non-overlapping** halves,
                and applies Spearman-Brown prediction formula.  This
                typically gives the low bound of the internal consistency.
            * 'spearman_brown_subsample_half': reconstructs two sets of
                data by two independent subsamplings of the original data
                into half (without replacement), computes the consistency
                between two sets (which can potentially share some
                **overlapping trials**), and applies Spearman-Brown formula.
                Empirically, this gives similar values as
                'bootstrap_assume_true_ctr' does, BUT DOES NOT HAVE
                CLEAR MATHEMATICAL FOUNDATION.  USE WITH CARE.
            * 'spearman_brown_subsample_half_replacement': Same as
                'spearman_brown_subsample_half' but this subsamples trials
                with **replacement** as in typical bootstrapping.
                This is essentially making bootstrap replicas of
                the original data and running 'spearman_brown_split_half'
                over those replicas.  As expected, this gives very
                similar internal consistency values as 'bootstrap' does.
    Nctr: array-like, shape=(# of images, # of neurons), default=None
        If given, ``Nctr`` will be used as the true central tendency values
        of neuronal responses to images (across trials).  Otherwise,
        the mean across trials will be computed and used by default.
    sync: bool, default=True
        If True, approximate time synchrony across images will be maintained.
        (aka non-shuffled)
    center: callable, default=np.mean
        The function used to estimate the central tendency of responses
        to images for a given neuron across (reconstructed) trials.  This must
        accept the keyword (named) argument 'axis' like np.mean().
    summary_center: callable, or 'raw', default=np.mean
        The function used to estimate the central tendency across different
        reconstructions of the data (e.g., bootstrapping samples).  If 'raw' is
        given, the raw values will be returned.
    summary_spread: callable, default=np.std
        The function used to estimate the spread across different
        reconstructions of the data.
    n_iter: int
        The # of reconstructions of the original data (e.g., bootstrap
        samplings, different split-halves).

    Returns
    -------
    If 'summary_center' is not 'raw':
        r: array-like, shape=(# of neurons)
            Contains each neuron's estimated self-self consistency across
            the given images.
        s: array-like, shape=(# of neurons)
            Spread of self-self consistencies of neurons.
    Else:
        rs: array-like, shape=(# of neurons, ``n_iter``)
            Contains each neuron's estimated self-self consistency across
            the given images over different reconstructions of the data.
    """
    n_img = 0
    n_neu = N[0].shape[-1]

    for N0 in N:
        assert N0.ndim == 3
        assert N0.shape[-1] == n_neu
        n_img += N0.shape[1]

    if Nctr is None:
        Nctr = [center(N0, axis=0) for N0 in N]
        Nctr = np.row_stack(Nctr)

    assert Nctr.shape == (n_img, n_neu)
    # check mode
    assert mode in ['bootstrap', 'bootstrap_assume_true_ctr',
            'spearman_brown_split_half',
            'spearman_brown_subsample_half',
            'spearman_brown_subsample_half_replacement']
    # check metric
    if metric in ['spearman', 'spearmanr', 'pearson', 'pearsonr']:
        pass
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        pass
    elif callable(metric):
        pass
    elif hasattr(distance, metric):
        pass
    else:
        raise ValueError('Unknown "metric"')

    if 'spearman_brown' in mode and metric not in ['pearson', 'pearsonr']:
        from warnings import warn
        warn('Using Spearman-Brown prediction formula for metrics other than' \
                "Pearson's correlation coefficient is NOT intended, " \
                'and therefore the returned internal consistencies' \
                'WILL BE MATHEMATICALLY UNGROUNDED.'
                )

    # number crunching!!!
    results = Parallel(n_jobs=n_jobs)(delayed(_noise_estimation_worker)
            ([N0[:, :, ni] for N0 in N], Nctr[:, ni], metric, mode,
                center, summary_center, summary_spread,
                sync=sync, n_iter=n_iter) for ni in range(n_neu))
    results = np.array(results)

    # if distributions are requested... (undocumented!!)
    if summary_center == 'raw':
        return results

    # ...otherwise do the regular jobs
    r = results[:, 0]
    s = results[:, 1]

    return r, s


def _spearmanr_helper(x, y):
    return stats.spearmanr(x, y)[0]


def _noise_estimation_worker(N, nctr, metric, mode,
        center, summary_center, summary_spread, seed=0,
        n_iter=DEF_BITER, sync=True):
    """Helper function for noise_estimation().

    N: list of one neuron's responses. Each element's shape =
        (# of reps, # of images)
    nctr: the central tendencies of the neuron's responses
    """
    rng = np.random.RandomState(seed)
    n_img = len(nctr)
    if mode in ['bootstrap', 'spearman_brown_subsample_half',
            'spearman_brown_subsample_half_replacement']:
        # only this amount is needed because
        # "spearman_brown_subsample_half" and
        # "spearman_brown_subsample_half_replacement"
        # are computed in pair-wise fashion
        n_iter = int(np.sqrt(n_iter))
    assert n_iter > 1

    # -- deal with metric and mode
    # various "correctors"
    def corrector_dummy(X):
        return X

    def corrector_pearsonr(X):
        X = 1. - X
        return X

    corrector = corrector_dummy   # do nothing by default

    if metric in ['spearman', 'spearmanr']:
        metric = _spearmanr_helper   # but this will be slow.
    elif metric in ['pearson', 'pearsonr']:
        # NOTE: Pearson r will be computed by sklearn's
        # pairwise_distances() with metric="correlation" for
        # efficiency.  Because "correlation" is 1 - pearsonr,
        # the retunred values **MUST** be corrected as below.
        metric = 'correlation'
        corrector = corrector_pearsonr

    if mode == 'spearman_brown_split_half' and type(metric) is str:
        if metric in PAIRWISE_DISTANCE_FUNCTIONS:
            metric = PAIRWISE_DISTANCE_FUNCTIONS[metric]
        elif not callable(metric):
            metric = getattr(distance, metric)

    # -- various subsampling helper functions
    def bsample_sync(M, n_div=1):
        n_rep, _ = M.shape
        ri = rng.randint(n_rep, size=n_rep / n_div)
        return center(M[ri], axis=0), []   # should be 1D vector

    def bsample_async(M, n_div=1):
        n_rep, _ = M.shape
        # x = [center(e[rng.randint(n_rep, size=n_rep)]) for e in M.T]
        x = [e[rng.randint(n_rep, size=n_rep / n_div)] for e in M.T]
        x = np.array(x).T   # in (# reps, # imgs)
        return center(x, axis=0), []

    def bsamplehalf_sync(M):
        return bsample_sync(M, n_div=2)[0], bsample_sync(M, n_div=2)[0]

    def bsamplehalf_async(M):
        return bsample_async(M, n_div=2)[0], bsample_async(M, n_div=2)[0]

    def bsamplefull_sync(M):
        return bsample_sync(M)[0], bsample_sync(M)[0]

    def bsamplefull_async(M):
        return bsample_async(M)[0], bsample_async(M)[0]

    def splithalf_sync(M):
        n_rep, _ = M.shape
        ri = range(n_rep)
        rng.shuffle(ri)   # without replacement
        ri1 = ri[:n_rep / 2]
        ri2 = ri[n_rep / 2:]
        return center(M[ri1], axis=0), center(M[ri2], axis=0)

    def splithalf_async(M):
        n_rep, _ = M.shape
        x = []
        y = []
        ri = range(n_rep)
        for e in M.T:   # image major
            rng.shuffle(ri)  # without replacement
            ri1 = ri[:n_rep / 2]
            ri2 = ri[n_rep / 2:]
            x.append(e[ri1])
            y.append(e[ri2])
        x = np.array(x).T   # in (# reps, # imgs)
        y = np.array(y).T
        return center(x, axis=0), center(y, axis=0)

    SAMPLE_FUNC_REGISTRY = {
            ('bootstrap', True): bsamplefull_sync,
            ('bootstrap', False): bsamplefull_async,
            ('bootstrap_assume_true_ctr', True): bsample_sync,
            ('bootstrap_assume_true_ctr', False): bsample_async,
            ('spearman_brown_split_half', True): splithalf_sync,
            ('spearman_brown_split_half', False): splithalf_async,
            ('spearman_brown_subsample_half', True): splithalf_sync,
            ('spearman_brown_subsample_half', False): splithalf_async,
            ('spearman_brown_subsample_half_replacement', True): bsamplehalf_sync,
            ('spearman_brown_subsample_half_replacement', False): bsamplehalf_async,
        }
    sample = SAMPLE_FUNC_REGISTRY[mode, sync]

    # -- reconstruct the data many n_iter times
    Xsmp = []
    Ysmp = []
    for _ in xrange(n_iter):
        xs = []
        ys = []
        for N0 in N:
            x, y = sample(N0)
            xs.extend(x)
            ys.extend(y)
        xs = np.array(xs)
        ys = np.array(ys)
        assert xs.shape == (n_img, )
        Xsmp.append(xs)
        Ysmp.append(ys)

    Xsmp = np.array(Xsmp)
    if mode == 'bootstrap_assume_true_ctr':
        Ysmp = [nctr]
    else:
        Ysmp = np.array(Ysmp)

    # -- get the numbers and done
    if mode != 'spearman_brown_split_half':
        ds = np.ravel(pairwise_distances(Ysmp, Xsmp, metric=metric))
    else:
        assert Xsmp.shape == Ysmp.shape
        # Essentially, this equals to taking the diagonalof the above.
        ds = np.array([metric(x, y) for x, y in zip(Xsmp, Ysmp)])
    ds = corrector(ds)

    if 'spearman_brown' in mode:
        ds = 2. * ds / (1. + ds)

    if summary_center == 'raw':
        return ds
    return summary_center(ds), summary_spread(ds)


def noise_estimation_per_img(N, metric='diff', mode='bootstrap',
        center=np.mean, summary_spread=np.std,
        n_jobs=1, n_iter=DEF_BITER):
    """Estimate the self-self consistencies of neurons for individual images.
    This is similar to noise_estimation().

    Paramters
    ---------
    N: list of array-like
        List of neuronal data.  Each item should be an array and should
        have the shape of (# of trials, # of images, # of neurons).
        Each item represents a subset of neuronal data for different
        images (e.g., V0, V3, V6).  While the # of trials can be different,
        the neurons must be the same across all items in ``N``.
    metric: string, default='diff'
        Which metric to use to compute the noise level.
            * 'diff': difference
            * 'absdiff': absolute difference (especially for MAD)
    mode: string, default='bootstrap'
        Which method to use to compute the noise level.
            * 'bootstrap': do bootstrapping
            * 'se': compute standard error (overrides metric, center, and
                summary_spread)
    center: callable, default=np.mean
        The function used to estimate the central tendency of responses
        to images for a given neuron across (reconstructed) trials.  This must
        accept the keyword (named) argument 'axis' like np.mean().
    summary_spread: callable, or 'raw', default=np.std
        The function used to estimate the spread across different
        reconstructions of the data (e.g., bootstrapping samples).  This must
        accept the keyword (named) argument 'axis' like np.mean().
        If 'raw' is given, the raw values will be returned.
    n_iter: int
        The # of reconstructions of the original data (e.g., bootstrap
        samplings).

    Returns
    -------
    If 'summary_spread' is not 'raw':
        r: array-like, shape=(# of neurons, # of images)
            Noise level for each image and neuron
    Else:
        rs: array-like, shape=(# of neurons, ``n_iter``, # of images)
            Noise level raw data for each image, neuron, and reconstruction
    """
    n_neu = N[0].shape[-1]

    for N0 in N:
        assert N0.ndim == 3
        assert N0.shape[-1] == n_neu

    if metric not in ['diff', 'absdiff']:
        raise ValueError('Not recognized "metric"')

    if mode not in ['bootstrap', 'se']:
        raise ValueError('Not recognized "mode"')

    # -- compute standard error
    if mode == 'se':
        results = []
        for N0 in N:
            n = N0.shape[0]
            s = np.std(N0, axis=0, ddof=1).T   # in (# of neurons, # of imgs)
            se = s / np.sqrt(n)
            results.append(se)
        results = np.concatenate(results, axis=1)
        return results

    # -- ...otherwise
    results = Parallel(n_jobs=n_jobs)(delayed(_noise_estimation_per_img_worker)
            ([N0[:, :, ni] for N0 in N], metric, mode,
                center, summary_spread,
                n_iter=n_iter) for ni in range(n_neu))
    results = np.array(results)
    return results


def _noise_estimation_per_img_worker(N, metric, mode, center,
        summary_spread, n_iter=DEF_BITER, seed=0):
    rng = np.random.RandomState(seed)
    n_iter = int(np.sqrt(n_iter))
    assert n_iter > 1

    # -- various subsampling helper functions copied from _noise_estimation_worker()
    # XXX: should they be standalone functions?
    def bsample_sync(M, n_div=1):
        n_rep, _ = M.shape
        ri = rng.randint(n_rep, size=n_rep / n_div)
        return center(M[ri], axis=0)   # should be 1D vector

    SAMPLE_FUNC_REGISTRY = {
            'bootstrap': bsample_sync
        }
    sample = SAMPLE_FUNC_REGISTRY[mode]

    # -- various metric functions can be placed here. but only "diff" as of now.
    def metric_diff(x1, x2):
        return x1 - x2

    def metric_absdiff(x1, x2):
        return np.abs(x1 - x2)

    METRIC_FUNC_REGISTRY = {
            'diff': metric_diff,
            'absdiff': metric_absdiff,
        }
    metric = METRIC_FUNC_REGISTRY[metric]

    # -- reconstruct the data many n_iter times
    Xsmp = []   # shape = (n_iter, # of images)
    Ysmp = []
    for _ in xrange(n_iter):
        xs = []
        ys = []
        for N0 in N:
            x = sample(N0)
            y = sample(N0)
            xs.extend(x)
            ys.extend(y)
        xs = np.array(xs)
        ys = np.array(ys)
        # assert xs.shape == (n_img, )
        Xsmp.append(xs)
        Ysmp.append(ys)

    ds = []   # shape == (n_iter, n_img)
    # note that this is done at individual image level
    # so cannot be accelerated by something like
    # sklearn's pairwise_distances()
    for xs in Xsmp:
        for ys in Ysmp:
            ds.append(metric(xs, ys))

    if summary_spread == 'raw':
        return ds
    return summary_spread(ds, axis=0)


def get_subset_splits(meta, npc_train, npc_tests, num_splits,
                      catfunc, train_q=None, test_qs=None, test_names=None, npc_validate=0):
        """Create train/test/validate splits of the data in a standard way.

        NB:  The seed for the random number generator for generating the random
        splits is hard-code to a fixed value (namely, 0) in this function.

        meta: metadata to split

        npc_train: (int) number of images per grouping to be used for training

        npc_tests: (list of int) numbers of images per grouping to be used for testing, one for each test query

        num_splits: (int) number of splits to generate

        catfunc: (callable with one variable or None) function used to produce labels for each metadata record
        relative to which the balanced grouping should be done.  If no balancing, set this to None.

        train_q: query that is applied to the metadata to restrict what samples can be used during training. This can be:
                     i. dictionary: [(key, [value list]), (key, [value list]) .... ] such that the samples are subsetted to have metadata[key] be in [value list] for all the keys/value lists in the dictionary.  For example:
                                 {'category': ['bike', 'car'], 'var': ['V6']}  means that only records in which the category variable is either 'bike' or 'car' and in which the 'var' variable is 'V6' are used during training.
                     ii. function: taking metadata records as single arguments are returning true or false, such that only the records that get "True" are used during training.  for example
                                 lambda x: x['var'] in ['V6'] and x['category'] in ['bike', 'car']   is effectively the as the dictionary spec described previously.
        test_q:  same as train_q, except applies to the testing splits.  Typically train_q == test_q, but doesn't have too

        test_names: (list of strings) Names of test splits

        npc_validate: number of images to hold out for validation.


        returns:  splits (list of num_splits dictionaries of arrays of metadata indices,
                            including one entry for training split and one for each test split),
                  validations (list of arrays of metadata indices withheld for all train/test splits)
        """
        if train_q is not None:
            train_q = get_lambda_from_query_config(train_q)
        if test_qs is not None:
            test_qs = map(get_lambda_from_query_config, test_qs)

        assert (npc_train is None) or (isinstance(npc_train, int) and npc_train > 0), 'npc_train must be None or an integer greater than 0, not: %s' % repr(npc_train)

        train_inds = np.arange(len(meta)).astype(np.int)
        if test_qs is None:
            test_qs = [test_qs]
        if test_names is None:
            assert len(test_qs) == 1
            test_names = ['test']
        else:
            assert len(test_names) == len(test_qs)
            assert 'train' not in test_names
        test_ind_list = [np.arange(len(meta)).astype(np.int) for _ in range(len(test_qs))]
        if train_q is not None:
            sub = np.array(map(train_q, meta)).astype(np.bool)
            train_inds = train_inds[sub]
        for _ind, test_q in enumerate(test_qs):
            if test_q is not None:
                sub = np.array(map(test_q, meta)).astype(np.bool)
                test_ind_list[_ind] = test_ind_list[_ind][sub]

        all_test_inds = list(itertools.chain(*test_ind_list))
        all_inds = np.sort(np.unique(train_inds.tolist() + all_test_inds))
        categories = np.array(map(catfunc, meta))
        ucategories = np.unique(categories[all_inds])
        utestcategorylist = [np.unique(categories[_t]) for _t in test_ind_list]
        utraincategories = np.unique(categories[train_inds])
        rng = np.random.RandomState(0)  # or do you want control over the seed?
        splits = [dict([('train', [])] + [(tn, []) for tn in test_names]) for _ in range(num_splits)]
        validations = [[] for _ in range(len(test_qs))]
        for cat in ucategories:
            cat_validates = []
            ctils = []
            for _ind, test_inds in enumerate(test_ind_list):
                cat_test_inds = test_inds[categories[test_inds] == cat]
                ctils.append(len(cat_test_inds))
                if npc_validate > 0:
                    assert len(cat_test_inds) >= npc_validate, 'not enough to validate'
                    pv = rng.permutation(len(cat_test_inds))
                    cat_validate = cat_test_inds[pv[:npc_validate]]
                    validations[_ind] += cat_validate.tolist()
                else:
                    cat_validate = []
                cat_validates.extend(cat_validate)
            cat_validates = np.sort(np.unique(cat_validates))
            for split_ind in range(num_splits):
                cat_train_inds = train_inds[categories[train_inds] == cat]
                if len(cat_train_inds) < np.mean(ctils):
                    cat_train_inds = train_inds[categories[train_inds] == cat]
                    cat_train_inds = np.array(list(set(cat_train_inds).difference(cat_validates)))
                    if cat in utraincategories:
                        assert len(cat_train_inds) >= npc_train, 'not enough train for %s, %d, %d' \
                                                                 % (cat, len(cat_train_inds), npc_train)
                    cat_train_inds.sort()
                    p = rng.permutation(len(cat_train_inds))
                    if npc_train is not None:
                        npct = npc_train
                    else:
                        npct = len(cat_train_inds)
                    cat_train_inds_split = cat_train_inds[p[:npct]]
                    splits[split_ind]['train'] += cat_train_inds_split.tolist()
                    for _ind, (test_inds, utc) in enumerate(zip(test_ind_list, utestcategorylist)):
                        npc_test = npc_tests[_ind]
                        cat_test_inds = test_inds[categories[test_inds] == cat]
                        cat_test_inds_c = np.array(list(
                            set(cat_test_inds).difference(cat_train_inds_split).difference(cat_validates)))
                        if cat in utc:
                            assert len(cat_test_inds_c) >= npc_test, 'not enough test for %s %d %d' \
                                                                     % (cat, len(cat_test_inds_c), npc_test)
                        p = rng.permutation(len(cat_test_inds_c))
                        cat_test_inds_split = cat_test_inds_c[p[: npc_test]]
                        name = test_names[_ind]
                        splits[split_ind][name] += cat_test_inds_split.tolist()
                else:
                    all_cat_test_inds = []
                    for _ind, (test_inds, utc) in enumerate(zip(test_ind_list, utestcategorylist)):
                        npc_test = npc_tests[_ind]
                        cat_test_inds = test_inds[categories[test_inds] == cat]
                        cat_test_inds_c = np.sort(np.array(list(
                            set(cat_test_inds).difference(cat_validates))))
                        if cat in utc:
                            assert len(cat_test_inds_c) >= npc_test, 'not enough test for %s %d %d' \
                                                                     % (cat, len(cat_test_inds_c), npc_test)
                        p = rng.permutation(len(cat_test_inds_c))
                        cat_test_inds_split = cat_test_inds_c[p[: npc_test]]
                        name = test_names[_ind]
                        splits[split_ind][name] += cat_test_inds_split.tolist()
                        all_cat_test_inds.extend(cat_test_inds_split)
                    cat_train_inds = \
                        np.array(
                            list(
                                set(cat_train_inds).difference(all_cat_test_inds).difference(cat_validates)))
                    if cat in utraincategories:
                        assert len(cat_train_inds) >= npc_train, 'not enough train for %s, %d, %d' % (
                            cat, len(cat_train_inds), npc_train)
                    cat_train_inds.sort()
                    p = rng.permutation(len(cat_train_inds))
                    if npc_train is not None:
                        npct = npc_train
                    else:
                        npct = len(cat_train_inds)
                    cat_train_inds_split = cat_train_inds[p[:npct]]
                    splits[split_ind]['train'] += cat_train_inds_split.tolist()

        return splits, validations


def compute_metric(F, dataset, eval_config, return_splits=False, attach_models=False):
    """This is a wrapper function for compute_metric_base, which allows you to pass a
       dataset object instead of a metadata array directory.   The purpose of having this
       function is for legacy code, since this was the function that was originally
       written.   But compute_metric_base is more flexible, and is preferable for further
       use.
    """

    return compute_metric_base(F, dataset.meta, eval_config,
                               return_splits=return_splits,
                               attach_models=attach_models,
                               dataset=dataset)


def compute_metric_base(F, meta, eval_config, return_splits=False,
                        attach_models=False, dataset=None):
    """Unified interface for evaluating features via a supervised learning paradigm, involving cross-validated training
     of a feature representation with some type of predictor based on a labeled training set, testing the results on a
      testing set, and computing various numerical summary metrics of performance.

    ARGUMENTS:

    - F -- array, shape (# samples, # features).   Feature repesentation matrix
    - meta -- (record array, len = # samples) metadata tabular data array that contains
               information for creating splits as well as for getting labels for training
               and testing.
    - eval_config -- (dict) nested key-value specifying how do to the evaluation,
                     including how to make the splits, pick
                     the classifier/regressor/other predictor, and how to do the
                     training/testing.  Keys at the top level include:
              - train_q: query that is applied to the metadata to restrict what samples
                         can be used during training.
              This can be:
                     i. dictionary: [(key, [value list]), (key, [value list]) .... ] such that the samples are subsetted
                      to have metadata[key] be in [value list] for all the keys/value lists in the dictionary.
                      For example:
                                 {'category': ['bike', 'car'], 'var': ['V6']}  means that only records in which the
                                 category variable is either 'bike' or 'car' and in which the 'var' variable is 'V6'
                                 are used during training.
                     ii. function: taking metadata records as single arguments are returning true or false, such that
                     only the records that get "True" are used during training.  for example
                                 lambda x: x['var'] in ['V6'] and x['category'] in ['bike', 'car']   is effectively the
                                 same as the dictionary spec described previously.
              - test_q:  same as train_q, except applies to the testing splits.  Typically train_q == test_q, but
                doesn't have to.
              - npc_train: (integer) number of items per category used in training (see description of 'split_by'
                for what 'category' means)
              - npc_test:  (integer) number of items per category used in testing
              - npc_validate:  (integer)  number of items to entirely leave out of the category training and testing
                to reserve for some future validation setting.
              - num_splits: (integer) how many times to split
              - split_by: string (key in metadata) or None -- which key to use for defining "category" for the purposes
                of creating balanced splits.
                          If None, then no balancing is enforced -- data is merely selected.
              - labelfunc: defines the label for each metadata record.  this can be:
                          i. function:  which takes the whole metadata array as its single argument and returns
                            (Labels, Unique_labels) where Labels is an array of label values
                            (of same length as meta array) and unique_labels is the list of unique labels present in the
                             label array, in the order that is desired for reporting.   (this second thing is not that
                              important; it can be set to None for convenience with no ill effect).  For example:
                                lambda x: (np.sqrt(x['ty']**2 + x['tz']**2), None)
                              specifies "radius" label for a regression problem.
                          ii. string:  key in metadata record to use as labels.  (e.g. "category")
              -metric_screen:  (string).  Which type of metric to use.   Options:
                                'classifier' -- e.g. a scikit-learn style classifier.
                                'kanalysis', 'kanalysis_linear', 'ksurface' -- various kernel analysis metrics
                                'regression' -- scikit-learn style continuous-valued regressor
              - metric_kwargs: (dictionary) Arguments for the metric function. If metric == 'classifier', then this
                  dictionary can contain the following:
                                       'model_type':
                                            (string) which classifier model to use (e.g. 'MCC2', 'svm.SVC', 7c)
                                       'model_kwargs': (dict)  arguments to pass to the scikit model constructor.
                                       'fit_kwargs': (dict) arguments to pass to the scikit model during the calling
                                            of the fit method.
                                       'normalization': (bool)  whether or not to sphere the data before training.
                                       'trace_noramlize': (bool) whether or not to normalize the data feature-wise
                                        before training
                                       'margins': return the margins of the classifier as an argument or not.
                                       (the reason this is present is that margins can be a very large array,
                                       so it is often inefficient to copy it around)
               'metric_lbls' -- which labels to subset the results for.
                        (not used that much.  maybe should be deprecated).  if None, then it is ignored.
    - attach_models: (bool) whether to attach the model objects (one for each split) produced by the predictor training
        for use in additional prediction.  these model objects are usually very large and cannot be stored in
        JSON format (e.g. in  mongo DB) so you may or may not want to attach them.
    - return_splits: (bool) whether to attach the split indices actually used. This is for convenient verification
        purposes, in that you might not want to have to regenerate the splits (using the get_splits method).


    RETURNS: result (dict).  contains all the the result data.  this will be in somewhat different formats for each type
        of metric.   It will always be of the following form:
                {'splits': if return_splits=True, the splits are contained in this key,
                 'split_results': list: result dictionaries  for each split,
                 + some summary metrics which are specific to the type of metric being used. When metric=='classlfier',
                    these will include:
                        accbal_loss -- summary of balanced accuracy (over classes and splits)
                        dprime_loss -- summary of dprime values (over classes and splits)
                        acc_loss -- summary of accuracy values (over classes and splits)
                  When metric == 'regression', these will include:
                        rsquared_loss -- summary of rsquared loss over splits
                  When metric is a kanalsis metric, these include whatever the relevant summaries for those.
                  Typically these keys are referenced in the setup for a hyperopt experiment in terms of specifying
                  the loss function for the run.
                }
    """
    metric = eval_config['metric_screen']
    metric_kwargs = eval_config['metric_kwargs']

    if 'precomp_splits' in eval_config:
        splits = eval_config['precomp_splits']['splits']
        validations = eval_config['precomp_splits']['validations']
        labels = eval_config['precomp_splits']['labels']
        if 'uniq_labels' in eval_config['precomp_splits']:
            uniq_labels = eval_config['precomp_splits']['uniq_labels']
        else:
            uniq_labels = np.unique(labels)

    else:
        # -- load and split the data
        splits, validations = get_splits_from_eval_config(eval_config,
                                                dataset=dataset, meta=meta)
        labelfunc = get_labelfunc_from_config(eval_config['labelfunc'], meta)
        labels, uniq_labels = labelfunc(meta)
        assert labels.shape[0] == meta.shape[0]

    # If requested (not default), compute the metric with validation images.
    # This is done by manipulating ``splits`` as follows.
    if eval_config.get('use_validation'):
        all_train_test = np.unique([e for split in splits for e in split['train'] + split['test']])
        splits = [{'train': all_train_test, 'test': validations}]

    # default values
    metric_lbls = eval_config.get('metric_labels', None)

    result = {}
    if metric in ['kanalysis', 'kanalysis_linear', 'ksurface']:
        #if '_binary' not in eval_config['labelfunc']:
        #    raise ValueError('Invalid labelfunc for kanalysis')

        #all_inds = np.arange(len(dataset.meta)).astype(np.int)
        #sub_train_q = np.array(map(train_q, dataset.meta)).astype(np.bool)
        #sub_test_q = np.array(map(test_qs[0], dataset.meta)).astype(np.bool)   # XXX: assume only one test_q for now
        #sub = sub_train_q | sub_test_q
        #all_inds = list(set(all_inds[sub]) - set(validations[0]))

        split_results = []
        for split in splits:
            if metric in ['kanalysis', 'kanalysis_linear', 'ksurface']:
                all_inds = np.sort(np.unique(split['train'] + split['test']))
                uniq_labels = list(uniq_labels)
                ka_X = F[all_inds]
                # ka_T: (#all_inds, #metric_lbls) filled with 0 or 1.
                ka_T = labels[all_inds]
                #subset the labels as indicted, if metri_lbls is not None
                if metric_lbls is not None:
                    ka_T = ka_T[:, [uniq_labels.index(lbl) for lbl in metric_lbls]]

                if metric == 'kanalysis':
                    ka, var_max_idxs = kanalysis.kanalysis(ka_X, ka_T, **metric_kwargs)
                    ka_pred = None
                    ka_pred_corr = None
                elif metric == 'kanalysis_linear':
                    ka = kanalysis.kanalysis_linear(ka_X, ka_T, **metric_kwargs)
                    ka_pred = None
                    ka_pred_corr = None
                    var_max_idxs = None
                elif metric ==  metric == 'ksurface':
                    ka = kanalysis.ksurface(ka_X, ka_T, **metric_kwargs)
                    ka_pred = None
                    ka_pred_corr = None
                    var_max_idxs = None
            elif metric in ['kanalysis_predict', 'kanalysis_linear_predict']:
                ka_X = F[split['train']]
                ka_Xpred = F[split['test']]
                ka_T = labels[split['train']]
                ka_Tpred = labels[split['test']]
                assert metric_lbls is None
                if metric == 'kanalysis_predict':
                    ka, ka_pred, ka_pred_corr, var_max_idxs = \
                         kanalysis.kanalysis_predict(ka_X, ka_T,
                                            ka_Xpred, ka_Tpred, **metric_kwargs)
                elif metric == 'kanalysis_predict_linear':
                    ka, ka_pred, ka_pred_corr, var_max_idxs = \
                         kanalysis.kanalysis_predict_linear(ka_X, ka_T,
                                            ka_Xpred, ka_Tpred, **metric_kwargs)
            split_result = {'ka': ka,
                            'ka_pred': ka_pred,
                            'ka_pred_corr': ka_pred_corr,
                            'var_max_idxs': var_max_idxs}
            split_results.append(split_result)
        result['split_results'] = split_results

        #adding loss key
        #TODO:  it is not really understood how best to summarize these various
        #kernel analysis curves in a single number.  For the non-crossvalidated
        #predictive measure, it seems like mean is meaningful.   For the CV predicted versions
        #perhaps min of predicted curve is OK;  the mean value under the
        #pred_corr curve also seems meaningful.
        if metric in ['kanalysis', 'kanalysis_linear', 'ksurface']:
            lossval = np.mean([sr['ka'].mean() for sr in split_results])
        elif metric in ['kanalysis_predict', 'kanalysis_linear_predict']:
            mean_curve = np.array([sr['ka_pred'] for sr in split_results]).mean(0)
            lossval = mean_curve.min()
        result[metric + '_loss'] = lossval

    # -- .... or deal with classifiers
    elif metric in ['classifier']:
        metric_kwargs = copy.deepcopy(metric_kwargs)
        model_type = metric_kwargs.pop('model_type')
        fit_kwargs = metric_kwargs.pop('fit_kwargs', {})
        model_kwargs = metric_kwargs.pop('model_kwargs', {})
        normalization = metric_kwargs.pop('normalization', True)
        margins = metric_kwargs.pop('margins', False)
        trace_normalize = metric_kwargs.pop('trace_normalize', False)

        # train classifier
        split_results = []
        for split in splits:
            train_X = F[split['train']]
            test_X = F[split['test']]
            train_y = np.array(labels[split['train']])
            test_y = np.array(labels[split['test']])
            model, res = classifier.train_scikits((train_X, train_y),
                                     (test_X, test_y),
                                     model_type=model_type,
                                     fit_kwargs=fit_kwargs,
                                     model_kwargs=model_kwargs,
                                     normalization=normalization,
                                     margins=margins,
                                     trace_normalize=trace_normalize)
            if attach_models:
                res['model'] = model ###TODO:  put this in an attachment

            split_results.append(res)

        res_labelset = list(res['labelset'])
        if metric_lbls is not None:
            objinds = [res_labelset.index(lbl) for lbl in metric_lbls]
        else:
            #if metric_lbls is None, we'll keep all the entries, assuming
            #the metric produces a vector to be averaged over
            objinds = slice(None)

        # compute metrics
        compute_classifier_metrics(result, split_results, metric_kwargs, objinds, res_labelset, metric_lbls)

    elif metric in ['regression']:
        metric_kwargs = copy.deepcopy(metric_kwargs)
        model_type = metric_kwargs.pop('model_type')
        fit_kwargs = metric_kwargs.pop('fit_kwargs', {})
        model_kwargs = metric_kwargs.pop('model_kwargs', {})
        normalization = metric_kwargs.pop('normalization', False)
        margins = metric_kwargs.pop('margins', False)
        trace_normalize = metric_kwargs.pop('trace_normalize', False)
        regression = metric_kwargs.pop('regression', True)

        # train classifier
        cms = []
        split_results = []
        for split in splits:
            train_X = F[split['train']]
            test_X = F[split['test']]
            train_y = np.array(labels[split['train']])
            test_y = np.array(labels[split['test']])
            model, res = classifier.train_scikits((train_X, train_y),
                                     (test_X, test_y),
                                     model_type=model_type,
                                     fit_kwargs=fit_kwargs,
                                     model_kwargs=model_kwargs,
                                     normalization=normalization,
                                     margins=margins,
                                     regression=regression,
                                     trace_normalize=trace_normalize)
            if attach_models:
                res['model'] = model ###TODO:  put this in an attachment
            split_results.append(res)
        res_labelset = None
        result['split_results'] = split_results

        rs_loss = 1 - np.mean([s['test_rsquared'] for s in split_results])
        result['rsquared_loss'] = rs_loss
        result['rsquared_loss_stderror'] = np.std([1 - s['test_rsquared'] for s in split_results])

        result['corr_loss'] = np.mean([1 - s['test_corr'] for s in split_results])
        result['corr_loss_stderror'] = np.std([1 - s['test_corr'] for s in split_results])

        if 'test_multi_rsquared' in split_results[0]:
            mrs_array = np.array([s['test_multi_rsquared'] for s in split_results])
            mrs_array_loss = (1 - mrs_array).mean(0)
            mrs_array_loss_stderror = (1 - mrs_array).std(0)
            mrs_loss = (1 - np.median(mrs_array, 1)).mean(0)
            mrs_loss_stderror = (1 - np.median(mrs_array, 1)).std(0)
            result['multi_rsquared_array_loss'] = mrs_array_loss
            result['multi_rsquared_array_loss_stderror'] = mrs_array_loss_stderror
            result['multi_rsquared_loss'] = mrs_loss
            result['multi_rsquared_loss_stderror'] = mrs_loss_stderror
    else:
        raise ValueError('Invalid metric')

    if return_splits:
        result['splits'] = (splits, validations)

    return result


def compute_classifier_metrics(result, split_results, metric_kwargs, objinds, res_labelset, metric_lbls):
	cms = []
	for split_result in split_results:
		cms.append(split_result['test_cm'])
	cms = np.array(cms).swapaxes(0,1).swapaxes(1,2)

	# compute metrics
	dp = dprime_bangmetric(cms, **metric_kwargs)
	#dp_loss = min(1e5, -dp[objinds].mean())   #is this really right?  I'm sort of skeptical that this is a good criterion
	dp_loss = (1./2 - (np.arctan(dp) / np.pi))[objinds].mean()   #what about this instead...?
	dp_loss_stderror = (1./2 - (np.arctan(dp) / np.pi))[objinds].std()

	#
	dp_sym = symmetrized_dprime(cms, **metric_kwargs)
	dp_sym_loss = (1. - dp_sym[objinds]).mean()
	dp_sym_loss_stderror =  (1. - dp_sym[objinds]).std()

	dp_lin = dprime(cms, use_zs=False, population_avg='pop')
	dp_lin_loss = (1. - dp_lin[objinds]).mean()
	dp_lin_loss_stderror = (1. - dp_lin[objinds]).std()

	acc = performances_bangmetric(cms, **metric_kwargs)
	acc_loss = (1. - acc[objinds]).mean()
	acc_loss_stderror = (1. - acc[objinds]).std()

	multiacc = np.array([sr['test_accuracy'] for sr in split_results])
	multiacc_loss = (1. - multiacc/100.).mean()
	multiacc_loss_stderror = (1. - multiacc/100.).std()

	accbal = performances_bangmetric(cms, balanced=True, **metric_kwargs)
	accbal_loss = (1. - accbal[objinds]).mean()
	accbal_loss_stderror = (1. - accbal[objinds]).std()

	result['split_results'] = split_results
	result['result_summary'] = {'cms': cms,
								  'objinds': objinds if (metric_lbls is not None) else None,
								  'dprime': dp,
								  'acc': acc,
								  'accbal': accbal,
								  'dp_sym': dp_sym,
								  'labelset': res_labelset}
	result['dprime_loss'] = dp_loss
	result['dprime_loss_stderror'] = dp_loss_stderror
	result['acc_loss'] = acc_loss
	result['acc_loss_stderror'] = acc_loss_stderror
	result['accbal_loss'] = accbal_loss
	result['accbal_loss_stderror'] = accbal_loss_stderror
	result['dp_sym_loss'] = dp_sym_loss
	result['dp_sym_loss_stderror'] = dp_sym_loss_stderror
	result['dp_lin_loss'] = dp_lin_loss
	result['dp_lin_loss_stderror'] = dp_lin_loss_stderror
	result['multiacc_loss'] = multiacc_loss
	result['multiacc_loss_stderror'] = multiacc_loss_stderror


def get_splits_from_eval_config(eval_config, dataset, meta=None):
    """Convenience function that gives you the splits associated with a evaluation
    configuration
    """
    if meta is None:
        meta = dataset.meta
    npc_train = eval_config['npc_train']
    npc_test = eval_config['npc_test']
    npc_validate = eval_config['npc_validate']
    num_splits = eval_config['num_splits']

    if eval_config['split_by'] is not None:
        catfunc = lambda x: x[eval_config['split_by']]
    else:
        catfunc = lambda x: True

    split_args = {'npc_train': npc_train,
                  'npc_tests': [npc_test],
                  'num_splits': num_splits,
                  'catfunc': catfunc,
                  'train_q': eval_config.get('train_q'),
                  'test_qs': [eval_config.get('test_q')],
                  'test_names': ['test'],
                  'npc_validate': npc_validate}

    if hasattr(dataset, 'get_subset_splits'):
        splits, validations = dataset.get_subset_splits(**split_args)
    else:
        splits, validations = get_subset_splits(meta=meta, **split_args)

    return splits, validations


def get_CMbased_consistency(F, dataset, tasks, eval_configs):
    """Get confusion matrices of features and human subjects

    TODO/Thoughts
    -------------
    This function assumes everything is kosher, and only supports d'-based consistency for now.

    0. Need to add the imputation function for human CMs to support, for instance,
       4-way tasks... And perhaps, this should go into ``neural_datasets``.

    """

    #setup
    if hasattr(eval_configs, 'keys'):
        eval_configs = [copy.deepcopy(eval_configs) for _ in range(len(tasks))]
    else:
        assert len(eval_configs) == len(tasks)
        assert all([e['ctypes'] == eval_configs[0]['ctypes'] for e in eval_configs])
        eval_configs = [copy.deepcopy(e) for e in eval_configs]
    for ec in eval_configs:
        consis_types = ec.pop('ctypes')

    # -- get human confusion matrices
    # confusion mat shape = (ans, gnd truth, n_indv)
    # get human CM
    # XXX: Need "imputer" here... OR inside datasetl.human_confusion_mat_by_task
    cms_Hl = map(dataset.human_confusion_mat_by_task, tasks)
    cms_H, cms_l = zip(*cms_Hl)

    # -- get features' confusion matrices for tasks
    cms_F = []
    recs = []

    for task, eval_config, cmH, cml in zip(tasks, eval_configs, cms_H, cms_l):
        evalc = get_evalc(eval_config, task)
        evalc['metric_labels'] = None
        evalc['metric_screen'] = 'classifier'
        rec = compute_metric(F, dataset, evalc)
        cmF = rec['result_summary']['cms']
        labelset = rec['result_summary']['labelset']
        assert set(labelset) == set(cml), (set(labelset), set(cml))
        cmls = reorder_to(np.array(labelset), np.array(cml))
        cmF = cmF[cmls][:, cmls]
        assert cmF.shape[0] == cmF.shape[1] == cmH.shape[0] == cmH.shape[1]
        cms_F.append(cmF)

    # bookkeeping things...
    record = {}
    record['reference'] = {'cms_features': cms_F}
    record['consistency'] = {}
    record['consistency_data'] = {}

    # get all consistency types with the CMs
    for recname, typestr, typecfg in consis_types:
        kw = typecfg['kwargs']
        res = consistency_from_confusion_mat(cms_H, cms_F,
                metric=typestr, **kw)
        c, m1, m2, sc1, sc2, ce, m1e, m2e, sc1e, sc2e = res
        record['consistency'][recname] = c
        record['consistency_data'][recname] = {'consistency': c,
                                               'metric': (m1, m2),
                                               'self_consistency': (sc1, sc2),
                                               'consistency_error': ce,
                                               'metric_error': (m1e, m2e),
                                               'self_consistency_error': (sc1e, sc2e)}

    return record


def get_evalc(eval_config, task):
    evalc = copy.deepcopy(eval_config)
    for k in task:
        if k not in evalc:
            evalc[k] = task[k]
        #elif k in ['train_q', 'test_q']:
        #    evalc[k].update(task[k])
        else:
            raise ValueError, "Can't combine key %s" % k
    return evalc


# copied from new_new_bandits.py
def kfunc(k, x):
    U = np.sort(np.unique(x[k]))
    return x[k], U   # U is added for consistency with kfuncb


# copied from new_new_bandits.py
def kfuncb(k, x, U=None):
    if U is None:
        U = np.sort(np.unique(x[k]))
    else:
        assert set(U) == set(np.sort(np.unique(x[k])))
    return np.array([x[k] == u for u in U]).T, U


def get_labelfunc_from_config(q, meta):
    """turns a dictionary into a function that returns a label
    for each record of a metadata table
    """
    if hasattr(q, '__call__'):
        return q
    else:
        LABEL_REGISTRY = dict([(k, functools.partial(kfunc, k))
                              for k in meta.dtype.names])
        return LABEL_REGISTRY[q]


def get_lambda_from_query_config(q):
    """turns a dictionary specificying a mongo query (basically)
    into a lambda for subsetting a data table

    TODO: implement OR or not, etc.
    See: https://github.com/yamins81/devthor/commit/367d9e0714d5d89dc08c4e37d653d716c87b64be#commitcomment-1657582
    """
    if hasattr(q, '__call__'):
        return q
    elif q == None:
        return lambda x: True
    else:
        return lambda x:  all([x[k] in v for k, v in q.items()])   # per Dan's suggestion..

