import numpy as np
from numba import njit
from scipy.optimize import curve_fit
from typing import Callable, Optional
from scipy.odr import ODR, RealData, Model


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


@njit
def _check_zeros(arr: np.ndarray) -> list:
    indexes = []

    for i, val in enumerate(arr):
        if val == 0:
            indexes.append(i)

    return indexes


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
    nw: bool
        Force the use of a non-weighted fit.
        It is set to `False` by default.

    Returns
    ---
    An unpacked list with the estimates on parameters and their
    errors, in the following order: p1,p2,...,e1,e2,...
    """

    if len(x) != len(y) and len(x) != len(y_err):
        raise ValueError("The arrays don't have the same size")

    # remove zeros
    indexes = _check_zeros(y_err)

    x = np.delete(x, indexes)
    y = np.delete(y, indexes)
    y_err = np.delete(y_err, indexes)

    # get initial values for parameters
    p0 = kwargs.get("p0", None)

    # weighted or not
    nw = kwargs.get("nw", False)
    y_err = y_err if not nw else np.full(len(y_err), max(y_err))

    parameters, cov_matrix = curve_fit(f, x, y, sigma=y_err, p0=p0)

    errors = np.sqrt(np.diag(cov_matrix))

    # coefficient of determination
    R2 = cod(y, f(x, *parameters))

    # output
    if len(parameters) == len(names):
        if display:
            print("====================================================")
            print("=====             RESULTS FROM FIT             =====")
            print("====================================================\n")

            for i, name in enumerate(names):
                print(f"{f'Estimated value of {name}:':<35} {f'{parameters[i]:.4e}':>16}")
                print(f"{f'Error on estimated value of {name}:':<35} {f'{errors[i]:.1e}':>16} \n")

            print(f"{'R2:':<35} {np.round(R2, 3):>16} \n")

            print("====================================================\n")
    else:
        raise ValueError("Too many/few names for the variables.")

    return *parameters, *errors


def function_fit_odr(
    x: np.ndarray,
    y: np.ndarray,
    x_err: np.ndarray,
    y_err: np.ndarray,
    f: Callable[[list[float], float], float],
    p0: list[float],
    names: tuple[str, ...],
    display: Optional[bool] = True,
):
    """
    This function performs a functional fit on a set
    of (x,y) points given as inputs. It takes into
    account both the errors on the x-values and on
    the y-values.

    Parameters
    ---
    x: numpy.ndarray
        The array with the x-values.
    y: numpy.ndarray
        The array with the y-values.
    x_err: numpy.ndarray
        The array with the error on the x-values.
    y_err: numpy.ndarray
        The array with the error on the y-values.
    f: 1D-function
        The function used for fitting. It has to be
        defined as follows:
        >>> def f(B, x):
                B[0] + B[1]*x + B[2]*x**2

        with the parameters to be fitted as a list.
    p0: list
        The list with the initial guesses on the
        values of the fit parameters.
    names: tuple[str]
        The tuple with the names of the parameters
        of the fit function.

    Optional Parameters
    ---
    display: bool
        Whether to display the result or not.

    Returns
    ---
    An unpacked list with the estimates on parameters and their
    errors, in the following order: p1,p2,...,e1,e2,...
    """

    if len(x) != len(y) and len(x) != len(y_err) and len(x) != len(x_err):
        raise ValueError("The arrays don't have the same size")

    # remove zeros
    indexes = [_check_zeros(x_err), _check_zeros(y_err)]
    for _ in indexes:
        x = np.delete(x, _)
        y = np.delete(y, _)
        x_err = np.delete(x_err, _)
        y_err = np.delete(y_err, _)

    # define model
    model = Model(f)

    # define data to be fitted
    data = RealData(x, y, x_err, y_err)

    # define the odr object
    odr = ODR(data, model, beta0=p0)

    # run the regression
    output = odr.run()

    # get results
    parameters = output.beta
    errors = output.sd_beta

    # coefficient of determination
    R2 = cod(y, f(parameters, x))

    # output
    if len(parameters) == len(names):
        if display:
            print("====================================================")
            print("=====             RESULTS FROM FIT             =====")
            print("====================================================\n")

            for i, name in enumerate(names):
                print(f"{f'Estimated value of {name}:':<35} {f'{parameters[i]:.4e}':>16}")
                print(f"{f'Error on estimated value of {name}:':<35} {f'{errors[i]:.1e}':>16} \n")

            print(f"{'R2:':<35} {np.round(R2, 3):>16}")
            print(f"{'Chi2:':<35} {np.round(output.sum_square, 3):>16} \n")

            print("====================================================\n")
    else:
        raise ValueError("Too many/few names for the variables.")

    return *parameters, *errors, output.sum_square


if __name__ == "__main__":
    pass
