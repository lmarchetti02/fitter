import numpy as np
from numba import njit
from typing import Optional


@njit
def _is_all_equal(data: np.ndarray) -> bool:
    """
    This function check if an array is full of
    the same element.
    """
    comparison = data[0]
    for i in data:
        if i != comparison:
            return False

    return True


def _linear_fit_nw(x: np.ndarray, y: np.ndarray, y_err: float, display: bool):
    """ """

    # calculate fit parameters
    D = np.array([[len(x), x.sum()], [x.sum(), (x**2).sum()]])
    D_inv = np.linalg.inv(D)
    B = np.array([y.sum(), (x * y).sum()])

    parameters = np.dot(D_inv, B)
    errors = np.sqrt(np.array([D_inv[0][0], D_inv[1][1]]) * y_err)

    return parameters, errors


def linear_fit(x: np.ndarray, y: np.ndarray, y_err: np.ndarray, display: Optional[bool] = True):
    """ """

    parameters = None
    errors = None

    if not _is_all_equal(y_err):

        # weights
        w = 1 / y_err**2

        # calculate fit parameters
        D = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x**2).sum()]])
        D_inv = np.linalg.inv(D)
        B = np.array([(w * y).sum(), (w * x * y).sum()])

        parameters = np.dot(D_inv, B)
        errors = np.sqrt(np.array([D_inv[0][0], D_inv[1][1]]))
    else:
        parameters, errors = _linear_fit_nw(x, y, y_err[0], display)

    # correlation coefficient
    ssx = ((x - x.mean()) ** 2).sum()
    ssy = ((y - y.mean()) ** 2).sum()
    num = ((x - x.mean()) * (y - y.mean())).sum()

    r = num / np.sqrt(ssx * ssy)

    # standard error of the estimate
    see = np.sqrt(((1 - r**2) * ssy) / (len(x) - 2))

    if display:
        names = ["a", "b"]

        print("====================================================")
        print("=====             RESULTS FROM FIT             =====")
        print("====================================================\n")

        for i, name in enumerate(names):
            print(f"{f'Estimated value of {name}:':<35} {f'{parameters[i]:.4e}':>16}")
            print(f"{f'Error on estimated value of {name}:':<35} {f'{errors[i]:.1e}':>16} \n")

        print(f"{'Correlation coefficient:':<35} {np.round(r, 3):>16}")
        print(f"{'Standard error of the estimate:':<35} {see:>16.1e} \n")

        print("==================================================== \n")

    return *parameters, *errors, r
