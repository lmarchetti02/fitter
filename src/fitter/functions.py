import numpy as np
from scipy.optimize import curve_fit
from typing import Callable, Optional


def cod(y: np.ndarray, f: np.ndarray, n_params: Optional[int] = None) -> float:
    """
    This function computes the 'coefficient of determination'
    R^2, given a set of data points and their fitted values.

    Parameters
    ---
    y: numpy.array
        The array of data points.
    f: numpy.array
        The array of fitted values.

    Returns
    ---
    The coefficient of determination R^2.
    """

    ss_res = ((y - f) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()

    R2 = 1 - (ss_res / ss_tot)

    if n_params:
        R2 = 1 - (1 - R2) * ((len(y) - 1) / (len(y) - n_params - 1))

    return R2


def function_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    f: Callable[[], float],
    names: tuple[str, ...],
    display: Optional[bool] = True,
    **kwargs,
):
    """
    This function performs a functional fit on a set
    of (x,y) points given as inputs.

    Parameters
    ---
    x: numpy.ndarray
        The array with the x-values.
    y: numpy.ndarray
        The array with the y-values.
    y_err: numpy.ndarray
        The array with the error on the y-values.
    f: 1D-function
        The function used for fitting
    names: tuple[str]
        The tuple with the names of the parameters
        of the fit function.

    Optional Parameters
    ---
    display: bool
        Whether to display the result or not.

    Extra Parameters
    ---
    p0: list
        The list with the initial guesses on the
        values of the fit parameters.

    Returns
    ---
    An unpacked list with the estimates on parameters and their
    errors, in the following order: p1,p2,...,e1,e2,...
    """

    if len(x) != len(y) and len(x) != len(y_err):
        raise ValueError("The arrays don't have the same size")

    # get initial values for parameters
    p0 = kwargs.get("p0", None)

    parameters, cov_matrix = curve_fit(f, x, y, sigma=y_err, p0=p0)

    errors = np.sqrt(np.diag(cov_matrix))

    # coefficient of determination
    R2 = cod(y, f(x, *parameters))

    # output
    if len(parameters) == len(names):
        if display:
            print("----------------- RESULTS FROM FIT ----------------- \n")
            print(f"{'R2:':<35} {np.round(R2, 3):>16} \n")

            for i, name in enumerate(names):
                print(f"{f'Estimated value of {name}:':<35} {f'{parameters[i]:.4e}':>16}")
                print(f"{f'Error on estimated value of {name}:':<35} {f'{errors[i]:.1e}':>16} \n")

            print("---------------------------------------------------- \n")
    else:
        raise ValueError("Too many/few names for the variables.")

    return *parameters, *errors
