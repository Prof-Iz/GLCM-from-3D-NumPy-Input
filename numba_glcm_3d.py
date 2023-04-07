# %%
import numpy as np
from numba import njit

# %%
@njit(cache=True, parallel=True)
def iteration(levels, x, offset, x_max, y_max, z_max):
    results = np.zeros((levels, levels))  # initialise results error

    for i, v in np.ndenumerate(x):
        x_offset = i[0] + offset[0]
        y_offset = i[1] + offset[1]
        z_offset = i[2] + offset[2]

        if (x_offset >= x_max) or (y_offset >= y_max) or (z_offset >= z_max):
            # if offset out of boundary skip
            continue

        value_at_offset = x[x_offset, y_offset, z_offset]

        results[v, value_at_offset] += 1

    return results / levels**2



def numba_glcm_3d(x: np.ndarray, delta: tuple[int] = (1, 1, 1), d: int = 1):
    """Same algorithm as in glcm_3d file. Used Numbas in an internal function to speed up iteration.

    Args:
        x (np.ndarray): input array. 3D. dtype int
        delta (tuple[int], optional): Direction vector from pixel. Defaults to (1, 1, 1).
        d (int, optional): Distance to check for neighbouring channel. Defaults to 1.

    Raises:
        Exception: if input is not of type dint or is not 3D

    Returns:
        _type_: GLCM Matrix
    """

    if "int" not in x.dtype.__str__():
        raise Exception("Input should be of dtype Int")

    if len(x.shape) != 3:
        raise Exception("Input should be 3 dimensional")

    offset = (delta[0] * d, delta[1] * d, delta[2] * d)  # offset from each pixel

    x_max, y_max, z_max = x.shape  # boundary conditions during enumeration

    levels = x.max() + 1  # 0:1:n assume contn range of pixel values

    return iteration(levels, x, offset, x_max, y_max, z_max)


# %%
if __name__ == "__main__":
    test_array_2 = np.random.randint(0, 1600, (300, 300, 300))
    result = numba_glcm_3d(test_array_2, delta=(1, 0, 0))



# %%
