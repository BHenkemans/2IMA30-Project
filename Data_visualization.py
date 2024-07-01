import tifffile
import matplotlib.pyplot as plt
import numpy as np

# impMillie = tifffile.imread('Datasets\detrended.tiff')

# imp = impMillie

# base_points = imp[660].copy()
# print("min: " + str(np.min(base_points)) + " max: " + str(np.max(base_points)) + " mean: " + str(np.mean(base_points)))
# print(slice.shape)


# # From the slice, we want to plot the values of the slice for a given x value
# # We can do this by plotting the slice against the y values
# plt.title('Slice of the image at time 660, y = 1590')
# plt.plot(slice[1590])
# plt.show()

# # We can also plot the values of the slice for a given y value
# # We can do this by plotting the slice against the x values
# plt.title('Slice of the image at time 660, x = 1590')
# plt.plot(imp[660][80])
# plt.show()

# plt.title('Slice of the image at time 660')
# plt.imshow(imp[660], cmap='gray')
# plt.show()

data = np.load('data_arrays_high_water.npz')
nOfIslands = data['nOfIslands']
IslandsTotalArea = data['IslandsTotalArea']
IslandsAverageSizes = data['IslandsAverageSizes']
IslandsMedianSizes = data['IslandsMedianSizes']
IslandTotalVolume = data['IslandTotalVolume']
IslandAverageVolume = data['IslandAverageVolume']

x_values = list(range(50, 661, 10))
ticks = list(range(0, len(x_values)))

tick_spacing = 5
display_x_values = x_values[::tick_spacing]
display_ticks = ticks[::tick_spacing]

plt.plot(nOfIslands, label='Number of Islands')
plt.ylim(bottom=0)
plt.xticks(ticks = display_ticks,labels=display_x_values)
plt.title('Number of islands over time')
plt.xlabel('Time')
plt.savefig('VisualizationsHighWater\Islands.png', format='png', dpi=300)
plt.clf()

plt.plot(IslandsTotalArea, label='Total area of Islands')
plt.ylim(bottom=0)
plt.xticks(ticks = display_ticks,labels=display_x_values)
plt.title('Total area of islands over time')
plt.xlabel('Time')
plt.savefig('VisualizationsHighWater\TotalAreaIslands.png', format='png', dpi=300)
plt.clf()

plt.plot(IslandsAverageSizes, label='Average size of Islands')
plt.ylim(bottom=0)
plt.xticks(ticks = display_ticks,labels=display_x_values)
plt.title('Average size of islands over time')
plt.xlabel('Time')
plt.savefig('VisualizationsHighWater\AverageSizeIslands.png', format='png', dpi=300)
plt.clf()

plt.plot(IslandsMedianSizes, label='Median size of Islands')
plt.ylim(bottom=0)
plt.xticks(ticks = display_ticks,labels=display_x_values)
plt.title('Median size of islands over time')
plt.xlabel('Time')
plt.savefig('VisualizationsHighWater\MedianSizeIslands.png', format='png', dpi=300)
plt.clf()

plt.plot(IslandTotalVolume, label='Total volume of Islands')
plt.ylim(bottom=0)
plt.xticks(ticks = display_ticks,labels=display_x_values)
plt.title('Total volume of islands over time')
plt.xlabel('Time')
plt.savefig('VisualizationsHighWater\TotalVolumeIslands.png', format='png', dpi=300)
plt.clf()

plt.plot(IslandAverageVolume, label='Average Volume of Islands')
plt.ylim(bottom=0)
plt.xticks(ticks = display_ticks,labels=display_x_values)
plt.title('Average volume of islands over time')
plt.xlabel('Time')
plt.savefig('VisualizationsHighWater\AverageVolumeIslands.png', format='png', dpi=300)
plt.clf()