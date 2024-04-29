import tifffile
import matplotlib.pyplot as plt
import numpy as np

impMillie = tifffile.imread('detrended.tiff')

imp = impMillie / 1e6 - 20

time1 = imp[1].copy()
time600 = imp[660].copy()

print(time1.shape)
print(time1.min(), time1.max())
print(time600.min(), time600.max())

z = time600.copy()
# z = np.array([[x**2 + y**2 for x in range(20)] for y in range(20)])
x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

# show hight map in 2d
# plt.figure()
# plt.title('z as 2d heat map')
# p = plt.imshow(z, cmap='seismic')
# plt.colorbar(p)
# plt.show()


import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

def plot_heatmap_frame(data, figsize=(15,10)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap='terrain')

    fig.tight_layout()
    
    # Writing plot to memory for reading with PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Opening image with PIL and copying it
    with Image.open(buf) as im:
        frame = im.copy()
    
    # Explicitly close the figure to release memory
    plt.close(fig)
    
    return frame


# Load your data (assuming you already have this part)
# with open("animation_data.obj", "rb") as dataFile:
#     imp = pickle.load(dataFile)

# Create the first frame manually
im = plot_heatmap_frame(imp[0])

# Append the following frames and save as GIF
with io.BytesIO() as myGif: 
    im.save(myGif, format='GIF', save_all=True, append_images=[plot_heatmap_frame(imp[i]) for i in range(600, 650)], optimize=False, duration=8, loop=0)
    # Reset pointer position to start of GIF data
    myGif.seek(0)
    
    # Save the GIF to a file
    with open("training_weights_new.gif", "wb") as f:
        f.write(myGif.read())
