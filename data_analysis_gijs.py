import tifffile
import matplotlib.pyplot as plt
import numpy as np

im = tifffile.imread('detrended.tiff')

time1 = im[1].copy()
time600 = im[660].copy()

print(time1.shape)
print(time1.min(), time1.max())
print(time600.min(), time600.max())

z = time600.copy()
# z = np.array([[x**2 + y**2 for x in range(20)] for y in range(20)])
x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

# show hight map in 2d
plt.figure()
plt.title('z as 2d heat map')
p = plt.imshow(z, cmap='Greys')
plt.colorbar(p)
plt.show()