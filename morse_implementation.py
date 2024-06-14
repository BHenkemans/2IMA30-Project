import tifffile
import matplotlib.pyplot as plt
import numpy as np

impMillie = tifffile.imread('Datasets\detrended.tiff')

imp = impMillie

base_points = imp[0].copy()
data_points = imp[659].copy().T

IBL = base_points.min()

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
    return options[max_edge] + eps**2 *  options[opp_edge], "{:02b}".format(3 - max_edge)

# cell_values = np.zeros((1600-1, 160-1))
cell_values = np.empty((1600-1, 160-1),object)
for coord_x in range(1600-1):
    for coord_y in range(160-1):
        cell_values[coord_x][coord_y] = cell_value(coord_x, coord_y)

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
print("minima: " + str(np.count_nonzero(is_minimum))) # Geeft 5577



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

print("saddles: " + str(np.count_nonzero(horizontal_saddles) + np.count_nonzero(vertical_saddles)))


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

# STEP 5: Merge maxima if they are connected by a (non-diagonal) path that does not go below the IBL
# We will do this using breadth-first search from the maxima
# We will use a queue to keep track of the cells we still need to visit
# We will use a dictionary to keep track of the cells we have already visited

# In order to keep track of the amount of islands, we will create a copy of the is_maximum array
is_island = is_maximum.copy()

# If a maximum is below the IBL, we will not consider it part of an island
for x in range(1599):
    for y in range(159):
        if is_island[x][y]:
            if cell_values[x][y][0] < IBL:
                is_island[x][y] = 0

# We will now define a function that merges the maxima
def merge_maxima(start):
    queue = [start]
    visited = set([start])
    # We will now perform a breadth-first search
    while queue:
        x, y = queue.pop(0)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 1599 and 0 <= new_y < 159:
                # If the cell is not visited and the cell value is above the IBL, we will add it to the queue
                if (new_x, new_y) not in visited and cell_values[new_x][new_y][0] >= IBL:
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))
                    # If the cell is a maximum, we will 'merge' it with the start by setting it to 0
                    if is_maximum[new_x][new_y] == 1:
                        is_island[new_x][new_y] = 0

# For each maximum, we will merge it with all connected maxima
for x in range(1599):
    for y in range(159):
        if is_island[x][y] == 1:
            merge_maxima((x, y))

print("maxima: " + str(np.count_nonzero(is_maximum)))
print("islands: " + str(np.count_nonzero(is_island)))




#STEP X: Visualizing the gradient fields

# We now plot the resulting gradient vector field
width, height = 1600, 160
x_offset, y_offset = 0, 0

# Since pyplot draws from bottom to top, we first draw the cells for maxima
for x in range(x_offset, x_offset + width-1):
    for y in range(y_offset, y_offset + height-1):
        if cell_values[x][y][0] < IBL:
            plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color='lightblue', zorder=0)
        else:
            plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color='bisque', zorder=0)
        if is_island[x][y] == 1:
            plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color='darkseagreen', zorder=0)
        elif is_maximum[x][y] == 1:
            plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color='red', zorder=0)

# # We then draw the edges
# for x in range(x_offset, x_offset + width):
#     for y in range(y_offset, y_offset + height):
#         if x < x_offset + width - 1:
#             if horizontal_edges[x][y][2] == 1:
#                 plt.plot([x, x+1], [y, y], 'g', zorder=3)
#                 path = make_segment_around_saddle(x, y, 1)
#                 x_values, y_values = zip(*path)
#                 plt.plot(x_values, y_values, 'black', zorder=2)
#             else:
#                 plt.plot([x, x+1], [y, y], 'grey', zorder=1)
#         if y < y_offset + height - 1:
#             if vertical_edges[x][y][2] == 1:
#                 plt.plot([x, x], [y, y+1], 'g', zorder=3)
#                 path = make_segment_around_saddle(x, y, 0)
#                 x_values, y_values = zip(*path)
#                 plt.plot(x_values, y_values, 'black', zorder=2)
#             else:
#                 plt.plot([x, x], [y, y+1], 'grey', zorder = 1)

# # We then draw the nodes
# for x in range(x_offset, x_offset + width):
#     for y in range(y_offset, y_offset + height):
#         if is_minimum[x][y] == 1:
#             plt.plot(x, y, marker = 'o', color='blue', zorder=4)
#             #plt.text(x, y, str(data_points[x][y]), color='black', ha='center', va='center', zorder=3)
#         else:
#             plt.plot(x, y, marker = 'o', color='grey', zorder=4)
#             #plt.text(x, y, str(data_points[x][y]), color='black', ha='center', va='center', zorder=3)

# # We then draw the gradient vertex-edge pairs
# for x in range(x_offset, x_offset + width):
#     for y in range(y_offset, y_offset + height):
#         if gradient_pair_vertex_edge[x][y] != 0:
#             if gradient_pair_vertex_edge[x][y] == 1:
#                 plt.arrow(x, y, 0, -0.3, head_width=0.1, head_length=0.1, color = 'yellow', zorder=5)
#             elif gradient_pair_vertex_edge[x][y] == 2:
#                 plt.arrow(x, y, 0.3, 0, head_width=0.1, head_length=0.1, color = 'yellow', zorder=5)
#             elif gradient_pair_vertex_edge[x][y] == 3:
#                 plt.arrow(x, y, 0, 0.3, head_width=0.1, head_length=0.1, color = 'yellow', zorder=5)
#             elif gradient_pair_vertex_edge[x][y] == 4:
#                 plt.arrow(x, y, -0.3, 0, head_width=0.1, head_length=0.1, color = 'yellow', zorder=5)

# # We then draw the gradient edge-cell pairs, which uses a little bit of a weird system to determine the direction (see the function)
# for x in range(x_offset, x_offset + width-1):
#     for y in range(y_offset, y_offset + height-1):
#         if gradient_pair_edge_cell[x][y] != 0:
#             if gradient_pair_edge_cell[x][y] == 1:
#                 plt.arrow(x + 0.5, y + 1, 0, -0.3, head_width=0.1, head_length=0.1, color = 'purple', zorder=5)
#             elif gradient_pair_edge_cell[x][y] == 2:
#                 plt.arrow(x + 0.5, y, 0, 0.3, head_width=0.1, head_length=0.1, color = 'purple', zorder=5)
#             elif gradient_pair_edge_cell[x][y] == 3:
#                 plt.arrow(x + 1, y + 0.5, -0.3, 0, head_width=0.1, head_length=0.1, color = 'purple', zorder=5)
#             elif gradient_pair_edge_cell[x][y] == 4:
#                 plt.arrow(x, y + 0.5, 0.3, 0, head_width=0.1, head_length=0.1, color = 'purple', zorder=5)

#plt.imshow(z.T, cmap='terrain')

# Since our data has the origin in the top left, we invert the y-axis
plt.gca().invert_yaxis()
# We set the background color to lightgrey for better visibility
plt.gca().set_facecolor('lightgrey')
# We set the aspect ratio to be equal
plt.axis('equal')
plt.savefig('islandsLowerRes.png', format='png', dpi=3000)
