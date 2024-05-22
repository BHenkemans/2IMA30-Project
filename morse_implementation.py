import tifffile
import matplotlib.pyplot as plt
import numpy as np

impMillie = tifffile.imread('detrended.tiff')

imp = impMillie

base_points = imp[0].copy()
data_points = imp[659].copy().T

print(data_points.shape)

IBL = base_points.min()
print(data_points.min(), data_points.max())

z = data_points.copy()
# z = np.array([[x**2 + y**2 for x in range(20)] for y in range(20)])
x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

def edge_value(coord_x, coord_y, dir, eps=1e-8):
    """determines morse func value of edge with 
    direction True == Right (and false is downwards)
    """
    point = data_points[coord_x][coord_y]
    if dir:
        second_point = data_points[coord_x+1][coord_y]
    else:
        second_point = data_points[coord_x][coord_y+1]
    return max(point, second_point) + eps*min(point, second_point)

print(data_points[10][15], data_points[10][16], data_points[11][15])
print(edge_value(10,15,True))

# horizontal_edges = np.array([edge_value(coord_x, coord_y, 1) for coord_x, coord_y in zip(np.arange(1600-1), np.arange(160-1))])
horizontal_edges = np.zeros((1600-1, 160))
for coord_x in range(1600-1):
    for coord_y in range(160):
        horizontal_edges[coord_x][coord_y] = edge_value(coord_x, coord_y, 1)

# vertical_edges = [edge_value(coord_x, coord_y, 0) for coord_x, coord_y in zip(np.arange(1600-1), np.arange(160-1))]
vertical_edges = np.zeros((1600, 160-1))
for coord_x in range(1600):
    for coord_y in range(160-1):
        vertical_edges[coord_x][coord_y] = edge_value(coord_x, coord_y, 0)

print(horizontal_edges[10][15])

def cell_value(coord_x, coord_y, eps = 1e-8):
    """Determines cell value based on edges around it
    Max edge value + eps^2 * opposite (not min)
    """
    hor_edges = horizontal_edges[coord_x][coord_y], horizontal_edges[coord_x][coord_y+1]
    ver_edges = vertical_edges[coord_x][coord_y], vertical_edges[coord_x+1][coord_y]

    options = hor_edges + ver_edges
    max_edge = np.argmax(options)
    if max_edge % 2 == 0:
        min_edge = max_edge + 1
    else:
        min_edge = max_edge - 1
    return options[max_edge] + eps**2 *  options[min_edge]

cell_values = np.zeros((1600-1, 160-1))
for coord_x in range(1600-1):
    for coord_y in range(160-1):
        cell_values[coord_x][coord_y] = cell_value(coord_x, coord_y)

print(cell_values[0][0])
print(np.min(horizontal_edges), np.min(vertical_edges), np.min(cell_values))