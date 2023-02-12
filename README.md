# Generating GLCM for a 3D Input

Grey Level Co-occurrence Matrices (GLCMs) is a statistical texture analysis method used to **characterize the texture of an image**. It involves the creation of a matrix that describes the statistical relationship between pairs of pixels that have a specified spatial relationship and intensity value (grey level).

*SkImage* already has a function `graycomatrix` that generates the GLCM based on 2D input. However, it does not work for 3D.

Thus I coded out the following function in numpy for 3D inputs. Intially this was for analysing 3D texture features for my [Final Year Project](https://github.com/Prof-Iz/3D-Segmentation-of-Glioblastoma-from-MRI)

If you have any suggestions or find issues feel free to make a pull request!

---
### Step by step

Refer to [python file](https://github.com/Prof-Iz/GLCM-from-3D-NumPy-Input/blob/master/glcm_3d.py) for full code. Below is just explanation of my code.

```python
glcm_3d(input: np.ndarray, delta: tuple[int] = (1, 1, 1), d: int = 1):
```
`input` is a 3D `numpy ndarray` of `dtype` **int**.

`delta` refers to offset to check for adjacent pixel. It is a tuple of form `(x,y,z)` where $(\Delta x, \Delta y \Delta z)$ from a given pixel. 


Since the voxel is in 3D space the angle of the neighbour may be in 26 possible directions based on the offset.


`d` is distance to check for.


```python
 x_max, y_max, z_max = input.shape  # boundary conditions during enumeration
```

As GLCM checks for pixels at an offset, to avoid error during indexing, maximum bounds of input tensor are noted down.

```python
levels = input.max() + 1 
```
Assume continuos integer range for possible pixel values in input array. Hence it is set as maximum.

```python
results = np.zeros((levels, levels))
```

Results are stored and computed in the array. It is initialised above.



```python
for i, v in np.ndenumerate(input):

        x_offset = i[0] + offset[0]

        y_offset = i[1] + offset[1]

        z_offset = i[2] + offset[2]
```

The input array is enumerated per voxel. For each one the position it should check for the neighbouring voxel is computed.

```python
if (x_offset >= x_max) 
or (y_offset >= y_max) 
or (z_offset >= z_max):
      # if offset out of boundary skip

      continue
```

If the neighbour of that voxel is out of bounds, that computation is skipped.

```python
value_at_offset = input[x_offset, y_offset, z_offset]
results[v, value_at_offset] += 1
```

The value of the voxel in the neighbouring position is retrieved. Taking the original pixel as `i` and the neighbour as `j`, the result matrix is incremented by 1 `P(i,j) += 1`

```python
return results / levels**2
```

Lastly the `2 dims` Matrix is returned after being divided by the square of levels to fulfill the formula requirements for

$$\frac{1}{N}\cdot P(i,j)$$
where $N$ is the number of possible pairs ($levels^2$)

