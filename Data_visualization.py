import tifffile
import matplotlib.pyplot as plt
import numpy as np

impMillie = tifffile.imread('Datasets\detrended.tiff')

imp = impMillie / 1e6 - 20

slice = imp[660].copy().T
print(slice.shape)


# From the slice, we want to plot the values of the slice for a given x value
# We can do this by plotting the slice against the y values
plt.title('Slice of the image at time 660, y = 1590')
plt.plot(slice[1590])
plt.show()

# We can also plot the values of the slice for a given y value
# We can do this by plotting the slice against the x values
plt.title('Slice of the image at time 660, x = 1590')
plt.plot(imp[660][80])
plt.show()

plt.title('Slice of the image at time 660')
plt.imshow(imp[660], cmap='gray')
plt.show()