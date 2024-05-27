import tifffile
import matplotlib.pyplot as plt
import numpy as np

impMillie = tifffile.imread('Datasets\detrended.tiff')

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
    is_max True => left/top vertex was maximal
    Last thing we return implies whether the edge is a saddle (1 indicates yes);
    This value is initialized as 1, but changed to 0 if a gradient pair with a vertex / cell is found
    """
    point = data_points[coord_x][coord_y]
    if dir:
        second_point = data_points[coord_x+1][coord_y]
    else:
        second_point = data_points[coord_x][coord_y+1]
    return [max(point, second_point) + eps*min(point, second_point), point >= second_point, 1]

print(data_points[10][15], data_points[10][16], data_points[11][15])
# print(edge_value(10,15,True))

# horizontal_edges = np.array([edge_value(coord_x, coord_y, 1) for coord_x, coord_y in zip(np.arange(1600-1), np.arange(160-1))])
# horizontal_edges = np.zeros((1600-1, 160))
horizontal_edges = np.empty((1600-1, 160),object)
for coord_x in range(1600-1):
    for coord_y in range(160):
        horizontal_edges[coord_x][coord_y] = edge_value(coord_x, coord_y, 1)

# vertical_edges = [edge_value(coord_x, coord_y, 0) for coord_x, coord_y in zip(np.arange(1600-1), np.arange(160-1))]
# vertical_edges = np.zeros((1600, 160-1))
vertical_edges = np.empty((1600, 160-1),object)
for coord_x in range(1600):
    for coord_y in range(160-1):
        vertical_edges[coord_x][coord_y] = edge_value(coord_x, coord_y, 0)

print(horizontal_edges[10][15])

def cell_value(coord_x, coord_y, eps = 1e-8):
    """Determines cell value based on edges around it
    Max edge value + eps^2 * opposite (not min)
    dir_max : 11 top, 10 bottom, 01 left, 00 right
    """
    hor_edges = horizontal_edges[coord_x][coord_y][0], horizontal_edges[coord_x][coord_y+1][0]
    ver_edges = vertical_edges[coord_x][coord_y][0], vertical_edges[coord_x+1][coord_y][0]

    options = hor_edges + ver_edges

    max_edge = np.argmax(options)
    if max_edge % 2 == 0:
        opp_edge = max_edge + 1
    else:
        opp_edge = max_edge - 1
    # if coord_y == 150 and coord_x == 1500:
    # print(max_edge, opp_edge, options)
    return options[max_edge] + eps**2 *  options[opp_edge], "{:02b}".format(3 - max_edge)

# cell_values = np.zeros((1600-1, 160-1))
cell_values = np.empty((1600-1, 160-1),object)
for coord_x in range(1600-1):
    for coord_y in range(160-1):
        cell_values[coord_x][coord_y] = cell_value(coord_x, coord_y)
        if coord_x == 1500 and coord_y == 100:
            print(horizontal_edges[coord_x][coord_y])

print(cell_values[0][0])
print(np.min(horizontal_edges), np.min(vertical_edges), np.min(cell_values))

is_minimum = np.ones((1600, 160))
gradient_pair_vertex_edge = np.empty((1600, 160),object)

def draw_vertex_edge_pair(coord_x, coord_y):
    """Determines for a vertex whether it is the maximum of an adjacent edge.
    If it is, determine for all edges of which it is a maximum, add a gradient pair to the smallest edge.
    0 => minimum (no larger edges), and 1 through 4 for clockwise pairs (starting top)
    """
    global horizontal_edges
    global vertical_edges
    global is_minimum

    gradient_so_far = 0
    gradient_min_value = np.inf

    # Define a list of conditions and corresponding actions
    conditions = [
        (coord_y > 0, (coord_x, coord_y-1), vertical_edges, 1, False),
        (coord_x < 1600-1, (coord_x, coord_y), horizontal_edges, 2, True),
        (coord_y < 160-1, (coord_x, coord_y), vertical_edges, 3, True),
        (coord_x > 0, (coord_x-1, coord_y), horizontal_edges, 4, False)
    ]

    for condition, coords, edge_array, gradient, edge_status in conditions:
        if condition:
            edge_value, edge_status_actual = edge_array[coords][0], edge_array[coords][1]
            if edge_status == edge_status_actual and edge_value < gradient_min_value:
                gradient_so_far = gradient
                gradient_min_value = edge_value

    if gradient_so_far > 0:
        is_minimum[coord_x][coord_y] = 0
        if gradient_so_far == 1:
            vertical_edges[coord_x][coord_y-1][2] = 0
        elif gradient_so_far == 2:
            horizontal_edges[coord_x][coord_y][2] = 0
        elif gradient_so_far == 3:
            vertical_edges[coord_x][coord_y][2] = 0
        elif gradient_so_far == 4:
            horizontal_edges[coord_x-1][coord_y][2] = 0

    return ((coord_x, coord_y), gradient_so_far)

for coord_x in range(1600):
    for coord_y in range(160):
        draw_vertex_edge_pair(coord_x, coord_y)
print(np.count_nonzero(is_minimum))