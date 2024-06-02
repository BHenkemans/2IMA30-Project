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

# STEP 1: Morse function values assignen
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

# STEP 2: gradient pairs assignen

is_minimum = np.ones((1600, 160))
horizontal_saddles = np.ones((1600-1, 160))
vertical_saddles = np.ones((1600, 160-1))
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
            vertical_saddles[coord_x][coord_y-1] = 0
        elif gradient_so_far == 2:
            horizontal_edges[coord_x][coord_y][2] = 0
            horizontal_saddles[coord_x][coord_y] = 0
        elif gradient_so_far == 3:
            vertical_edges[coord_x][coord_y][2] = 0
            vertical_saddles[coord_x][coord_y] = 0
        elif gradient_so_far == 4:
            horizontal_edges[coord_x-1][coord_y][2] = 0
            horizontal_saddles[coord_x-1][coord_y] = 0

    gradient_pair_vertex_edge[coord_x][coord_y] = gradient_so_far

    return ((coord_x, coord_y), gradient_so_far)

for coord_x in range(1600):
    for coord_y in range(160):
        draw_vertex_edge_pair(coord_x, coord_y)
print(np.count_nonzero(is_minimum)) # Geeft 5577



# We now want to determine the gradient pairs for the cells
is_maximum = np.ones((1600-1, 160-1))
# this is seen from the cell with coord_x coord_y
# note that this means the arrows need to be reversed in visualization
gradient_pair_edge_cell = np.zeros((1600-1, 160-1),object)

def draw_edge_cell_pair(array, coord_x, coord_y):
    """ Determines for an edge whether it is the maximum of an adjacent cell.
    If it is, determine for all cells of which it is a maximum, the smallest cell 
        and add a gradient pair to this edge and the smallest cell.
    0 => possible saddle (not a maximum of adjacent cell), and 1 through 2 for clockwise pairs (starting top/left)
    """
    
    # If we already know the edge is not a saddle by vertex-edge gradient pairs, we can skip this edge
    if array[coord_x][coord_y][2] == 0:
        return
    
    gradient_so_far = 0
    gradient_min_value = np.inf
    
    # Define a list of conditions and corresponding actions
    conditions = [
        (array is horizontal_edges and coord_y > 0, (coord_x, coord_y-1), "10", 1),
        (array is horizontal_edges and coord_y < 160-1, (coord_x, coord_y), "11", 2),
        (array is vertical_edges and coord_x > 0, (coord_x-1, coord_y), "00", 3),
        (array is vertical_edges and coord_x < 1600-1, (coord_x, coord_y), "01", 4)
    ]

    for condition, coords, cell_status_expected, gradient in conditions:
        if condition:
            cell_value, cell_status_actual = cell_values[coords]
            if cell_status_actual == cell_status_expected and cell_value < gradient_min_value:
                gradient_so_far = gradient
                gradient_min_value = cell_value
                
    if gradient_so_far > 0:
        if gradient_so_far == 1:
            horizontal_edges[coord_x][coord_y][2] = 0
            horizontal_saddles[coord_x][coord_y] = 0
            is_maximum[coord_x][coord_y-1] = 0
            gradient_pair_edge_cell[coord_x][coord_y-1] = gradient_so_far
        elif gradient_so_far == 2:
            horizontal_edges[coord_x][coord_y][2] = 0
            horizontal_saddles[coord_x][coord_y] = 0
            is_maximum[coord_x][coord_y] = 0
            gradient_pair_edge_cell[coord_x][coord_y] = gradient_so_far
        elif gradient_so_far == 3:
            vertical_edges[coord_x][coord_y][2] = 0
            vertical_saddles[coord_x][coord_y] = 0
            is_maximum[coord_x-1][coord_y] = 0
            gradient_pair_edge_cell[coord_x-1][coord_y] = gradient_so_far
        elif gradient_so_far == 4:
            vertical_edges[coord_x][coord_y][2] = 0
            vertical_saddles[coord_x][coord_y] = 0
            is_maximum[coord_x][coord_y] = 0
            gradient_pair_edge_cell[coord_x][coord_y] = gradient_so_far

for coord_x in range(1600):
    for coord_y in range(160):
        if coord_x != 1599:
            draw_edge_cell_pair(horizontal_edges, coord_x, coord_y)
        if coord_y != 159:
            draw_edge_cell_pair(vertical_edges, coord_x, coord_y)

print(np.count_nonzero(gradient_pair_edge_cell==0)) #geeft 11148
#print(np.count_nonzero(is_maximum)) # Geeft 11148
print(np.count_nonzero(horizontal_saddles) + np.count_nonzero(vertical_saddles))


# STEP 3 defining maxima, saddles, minima
# minimum: is_minimum[x][y] == 1
# saddle:   als horziontal_edges[x][y][2] == 1
#           als vertical_edges[x][y][2] == 1
# maxima: als is_maximum[x][y] == 1

# STEP 4: Make segments (between minima and saddles)
# a segment is a path from a saddle to a minimum
    # the path consists of a sequence of (x,y)-coordinates

def make_paths_from_saddle(coord_x, coord_y, dir):
    # path = np.empty((1,2), [])
    first_vertex = [coord_x, coord_y]
    if dir: 
        second_vertex = [coord_x+1, coord_y]
    else:
        second_vertex = [coord_x, coord_y+1]
    path = [[], []]
    for index, vertex in enumerate([first_vertex, second_vertex]):
        temp = [(vertex[0], vertex[1])]
        # path[index].append(vertex)
        # temp.append(vertex)
        # temp = np.append(temp, ([vertex[0], vertex[1]]))
        vertex_value = gradient_pair_vertex_edge[vertex[0]][vertex[1]]

        while vertex_value > 0:
            new_vertex = vertex.copy()
            if vertex_value == 1:
                new_vertex[1] -= 1
            elif vertex_value == 2:
                new_vertex[0] += 1
            elif vertex_value == 3:
                new_vertex[1] += 1
            else: 
                new_vertex[0] -= 1
            # temp.append(new_vertex)
            temp.append((new_vertex[0], new_vertex[1]))
            vertex = new_vertex.copy()
            vertex_value = gradient_pair_vertex_edge[vertex[0]][vertex[1]]
        path[index] = temp

    return (path[0], path[1])

def make_segment_around_saddle(coord_x, coord_y, dir):
    first_path, second_path = make_paths_from_saddle(coord_x, coord_y, dir)
    return list(reversed(first_path)) + second_path

# Known saddle check:
print(make_paths_from_saddle(582, 44, 1))
print(make_paths_from_saddle(1433, 47, 1))
print(make_paths_from_saddle(1591, 30, 1))

print(make_segment_around_saddle(1433, 47, 1))
print(np.count_nonzero(horizontal_saddles) + np.count_nonzero(vertical_saddles))

#print(is_minimum[:20, :20])

# We now plot the resulting gradient vector field
width, height = 50, 50

# Since pyplot draws from bottom to top, we first draw the cells for maxima
for x in range(width-1):
    for y in range(height-1):
        if is_maximum[x][y] == 1:
            plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color='red', zorder=1)

# We then draw the edges
for x in range(width):
    for y in range(height):
        if x < width - 1:
            if horizontal_edges[x][y][2] == 1:
                plt.plot([x, x+1], [y, y], 'g')
            else:
                plt.plot([x, x+1], [y, y], 'grey')
        if y < height - 1:
            if vertical_edges[x][y][2] == 1:
                plt.plot([x, x], [y, y+1], 'g')
            else:
                plt.plot([x, x], [y, y+1], 'grey')

# We then draw the nodes
for x in range(width):
    for y in range(height):
        if is_minimum[x][y] == 1:
            plt.plot(x, y, marker = 'o', color='blue', zorder=2)
        else:
            plt.plot(x, y, marker = 'o', color='grey', zorder=2)

# We then draw the gradient vertex-edge pairs
for x in range(width):
    for y in range(height):
        if gradient_pair_vertex_edge[x][y] != 0:
            if gradient_pair_vertex_edge[x][y] == 1:
                plt.arrow(x, y, 0, -0.3, head_width=0.1, head_length=0.1, color = 'yellow', zorder=3)
            elif gradient_pair_vertex_edge[x][y] == 2:
                plt.arrow(x, y, 0.3, 0, head_width=0.1, head_length=0.1, color = 'yellow', zorder=3)
            elif gradient_pair_vertex_edge[x][y] == 3:
                plt.arrow(x, y, 0, 0.3, head_width=0.1, head_length=0.1, color = 'yellow', zorder=3)
            elif gradient_pair_vertex_edge[x][y] == 4:
                plt.arrow(x, y, -0.3, 0, head_width=0.1, head_length=0.1, color = 'yellow', zorder=3)

# We then draw the gradient edge-cell pairs, which uses a little bit of a weird system to determine the direction (see the function)
for x in range(width-1):
    for y in range(height-1):
        if gradient_pair_edge_cell[x][y] != 0:
            if gradient_pair_edge_cell[x][y] == 1:
                plt.arrow(x + 0.5, y + 1, 0, -0.3, head_width=0.1, head_length=0.1, color = 'purple', zorder=3)
            elif gradient_pair_edge_cell[x][y] == 2:
                plt.arrow(x + 0.5, y, 0, 0.3, head_width=0.1, head_length=0.1, color = 'purple', zorder=3)
            elif gradient_pair_edge_cell[x][y] == 3:
                plt.arrow(x + 1, y + 0.5, -0.3, 0, head_width=0.1, head_length=0.1, color = 'purple', zorder=3)
            elif gradient_pair_edge_cell[x][y] == 4:
                plt.arrow(x, y + 0.5, 0.3, 0, head_width=0.1, head_length=0.1, color = 'purple', zorder=3)

#plt.imshow(z.T, cmap='terrain')

# Since our data has the origin in the top left, we invert the y-axis
plt.gca().invert_yaxis()
# We set the background color to lightgrey for better visibility
plt.gca().set_facecolor('lightgrey')
# We set the aspect ratio to be equal
plt.axis('equal')
plt.show()