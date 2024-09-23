import os
import sys
from collections import deque

from random import randint, random
from math import ceil
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import convolve


# def get_scaling_factors(shape):
#     max_dim = max(shape)  # Largest dimension for reference
#     scale_x = max_dim / shape[0]
#     scale_y = max_dim / shape[1]
#     scale_z = max_dim / shape[2]
#     return scale_x, scale_y, scale_z


def create_scale_cube_at_point(x, y, z):
    """
    Returns the vertices and faces for a unit cube centered at the given coordinates.
    """
    # Cube vertices (8 points)
    cube_vertices = np.array([
        [x - 0.5, y - 0.5, z - 0.5],
        [x + 0.5, y - 0.5, z - 0.5],
        [x + 0.5, y + 0.5, z - 0.5],
        [x - 0.5, y + 0.5, z - 0.5],
        [x - 0.5, y - 0.5, z + 0.5],
        [x + 0.5, y - 0.5, z + 0.5],
        [x + 0.5, y + 0.5, z + 0.5],
        [x - 0.5, y + 0.5, z + 0.5],
    ])

    # Cube faces (triangular faces defined by vertex indices)
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [1, 2, 6], [1, 6, 5],  # Right
        [2, 3, 7], [2, 7, 6],  # Back
        [3, 0, 4], [3, 4, 7],  # Left
    ])

    cube_edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ])

    return cube_vertices, cube_faces, cube_edges


def create_nested_numpy_array(sizes, current_dim=0, frequency=.1):
    """
    Recursively creates an N-dimensional NumPy array with varying sizes along each axis.

    Parameters:
    - sizes (list of int): A list where each element specifies the size for each dimension.
    - current_dim (int): The current dimension (used for recursion).

    Returns:
    - numpy.ndarray: A nested NumPy array with varying sizes.
    """
    # Base case: if we've reached the last dimension, return a 1D array of zeros and ones based on frequencies
    if current_dim == len(sizes) - 1:
        return np.random.choice([0, 1], size=sizes[current_dim], p=[frequency, 1 - frequency])
    
    # Recursively create arrays for each sub-dimension
    return np.array([create_nested_numpy_array(sizes, current_dim + 1, frequency) for _ in range(sizes[current_dim])])



def generate_random_hex():
    """
    idk I copied it from stack overflow
    """
    return "#%06x" % randint(0, 0xFFFFFF)
    # return "#{:06x}".format(randint(0, 0xFFFFFF))


class Ocean:
    def __init__(self, dimension_sizes=[10, 10], frequency=.1):
        self.ocean = self.create_ocean(dimension_sizes, frequency)


    def print_ocean(self):
        """
        Helper method to print out initial, unscanned oceans. Any node with a value of 0 is considered "land"

        Arguments:
            ocean: Nested arrays of integers
                Example: ([ [0, 1, 0, 0, 0, 1], ... ])
        """
        if len(self.dimension_sizes) == 3:
            values_flat = self.ocean.flatten()

            # Create lists to hold the cube vertices, face indices, and edges
            vertices_list = []
            faces_list = []
            edges_list = []
            colors = []

            # To keep track of the offset in indices when adding multiple cubes
            vertex_offset = 0



            # iterate over the flattened values
            for indx, val in enumerate(values_flat):
                # get the real coords of the flattened value index 
                coords = np.unravel_index(indx, self.ocean.shape)
                # check if this flattened value is `land``
                if val == 0:
                    # island_info is a dict where the keys are coordinates and the value is the island ID associated with the coords

                    color = generate_random_hex()
                    
                    cube_vertices, cube_faces, cube_edges = create_scale_cube_at_point(*coords)

                    # Add the vertices and faces for each cube
                    vertices_list.extend(cube_vertices)
                    faces_list.extend(cube_faces + vertex_offset)
                    edges_list.extend(cube_edges + vertex_offset)  # Track edges
                    colors.extend([color] * len(cube_faces))  # Add color for each face

                    vertex_offset += 8  # Each cube has 8 vertices

            # Convert to numpy arrays for mesh3d
            vertices_list = np.array(vertices_list)
            faces_list = np.array(faces_list)
            edges_list = np.array(edges_list)

            # Extract x, y, z coordinates from vertices
            x_vertices = vertices_list[:, 0]
            y_vertices = vertices_list[:, 1]
            z_vertices = vertices_list[:, 2]

            # Mesh3d plot for the cubes
            shape = self.ocean.shape

            mesh_fig = go.Figure(data=go.Mesh3d(
                x=x_vertices,
                y=y_vertices,
                z=z_vertices,
                i=faces_list[:, 0],
                j=faces_list[:, 1],
                k=faces_list[:, 2],
                facecolor=colors,  # Color for each face
                opacity=1,
            ))
            max_dim = max(shape)

            aspect_ratio = dict(
                x=shape[0] / max_dim,
                y=shape[1] / max_dim,
                z=shape[2] / max_dim,
            )

            mesh_fig.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=aspect_ratio
                )
            )
            # Scatter3d plot for the cube edges (outlines)
            x_edges = []
            y_edges = []
            z_edges = []

            # Extract edge points for plotting
            for edge in edges_list:
                for point in edge:  # Two points per edge
                    x_edges.append(vertices_list[point, 0])
                    y_edges.append(vertices_list[point, 1])
                    z_edges.append(vertices_list[point, 2])
                # Add None to break lines between edges
                x_edges.append(None)
                y_edges.append(None)
                z_edges.append(None)

            mesh_fig.add_trace(go.Scatter3d(
                x=x_edges,
                y=y_edges,
                z=z_edges,
                mode='lines',
                line=dict(color='black', width=50),
                showlegend=False
            ))
            
            mesh_fig.layout.scene.camera.projection.type = "orthographic"
            mesh_fig.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=aspect_ratio
                )
            )

            mesh_fig.show()
        elif len(self.dimension_sizes <= 2):
            print()
            print("HERE'S THE OCEAN:")
            print("*=============================================================*")
            for column in self.ocean:
                row = ''
                for node in column:
                    row += f" [ {'*' if not node else ' '} ]"
                print(row)
            print("*=============================================================*")
            print()
        else:
            print("I don't do visualizations for more than the 3rd dimension :(")


    def print_distinct_islands(self, island_info, num_island):
        """
        Given information related to given coordinates and associated islands, print out the fully scanned ocean with
        labeled islands.
        Note: It would be possible to construct this without the large(ish) ocean object and only the
        width/length of the ocean but I'm lazy and maybe I'll get to it one day.

        Arguments:
            island_info: Dictionary with keys of coordinates and values associated with the island that exists at that
            given coordinate. Island values will always be adjacent to coordinates with equal values 
                Example: { (0,0): 0, (10, 11): 1, (10, 12): 1, ... }

            ocean: Nested arrays of integers
                Example: ([ [0, 1, 0, 0, 0, 1], ... ])

            num_islands: Integer, total number of islands found within the ocean. Used for printing padding of 2d arrays.
        """
        # Visualization is hard... so I'm hard coding for specific dimensions
        if len(self.dimension_sizes) == 3:
            values_flat = self.ocean.flatten()

            # Dictionary to store the mapping of int to hex
            int_to_hex_map = {}

            # iterate over all of our island info data and generate a random hex value for each distinct island
            for num in island_info.values():
                if num not in int_to_hex_map:
                    int_to_hex_map[num] = generate_random_hex()

            # Create lists to hold the cube vertices, face indices, and edges
            vertices_list = []
            faces_list = []
            edges_list = []
            colors = []

            # To keep track of the offset in indices when adding multiple cubes
            vertex_offset = 0  

            # iterate over the flattened values
            for indx, val in enumerate(values_flat):
                # get the real coords of the flattened value index 
                coords = np.unravel_index(indx, self.ocean.shape)
                # check if this flattened value is `land``
                if val == 0:
                    # island_info is a dict where the keys are coordinates and the value is the island ID associated with the coords
                    island_id = island_info.get(coords)
                    # Grab the random hex value associated with the island
                    color = int_to_hex_map.get(island_id, 'magenta')
                    
                    # create a cube at that point
                    cube_vertices, cube_faces, cube_edges = create_scale_cube_at_point(*coords)

                    # Add the vertices, faces, edges, and colors for the cube to the lists
                    vertices_list.extend(cube_vertices)
                    faces_list.extend(cube_faces + vertex_offset)
                    edges_list.extend(cube_edges + vertex_offset)  # Track edges
                    colors.extend([color] * len(cube_faces))  # Add color for each face

                    # Each cube has 8 vertices so iterate the offset
                    vertex_offset += 8  

            # Convert to numpy arrays for mesh3d
            vertices_list = np.array(vertices_list)
            faces_list = np.array(faces_list)
            edges_list = np.array(edges_list)

            # Extract x, y, z coordinates from vertices
            x_vertices = vertices_list[:, 0]
            y_vertices = vertices_list[:, 1]
            z_vertices = vertices_list[:, 2]

            # Mesh3d plot for the cubes
            mesh_fig = go.Figure(data=go.Mesh3d(
                x=x_vertices,
                y=y_vertices,
                z=z_vertices,
                i=faces_list[:, 0],
                j=faces_list[:, 1],
                k=faces_list[:, 2],
                facecolor=colors,  # Color for each face
                opacity=1,
            ))

            # Scatter3d plot for the cube edges (outlines)
            x_edges = []
            y_edges = []
            z_edges = []

            # Extract edge points for plotting
            for edge in edges_list:
                for point in edge:  # Two points per edge
                    x_edges.append(vertices_list[point, 0])
                    y_edges.append(vertices_list[point, 1])
                    z_edges.append(vertices_list[point, 2])
                # Add None to break lines between edges
                x_edges.append(None)
                y_edges.append(None)
                z_edges.append(None)

            # Add traces to our 3D meshes
            mesh_fig.add_trace(go.Scatter3d(
                x=x_edges,
                y=y_edges,
                z=z_edges,
                mode='lines',
                line=dict(color='black', width=5),
                showlegend=False
            ))

            # Now that we've create dthe graph, scale the axes to ensure the shapes stay as cubes
            shape = self.ocean.shape
            max_dim = max(shape)
            aspect_ratio = dict(
                x=shape[0] / max_dim,
                y=shape[1] / max_dim,
                z=shape[2] / max_dim,
            )
            mesh_fig.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=aspect_ratio
                )
            )
            
            # Make the graph orthographic by default
            mesh_fig.layout.scene.camera.projection.type = "orthographic"
            mesh_fig.show()

        elif len(self.dimension_sizes <= 2):
            ################################################################################################################
                                                                # old

            # Maximum cell width is the number of digets of the last island found (ie island `10` = width of 2)
            maximum_cell_width = len(str(num_island))
            print()
            print("Results:")
            print("*======================================================================*")
            print()
            for x, column in enumerate(self.ocean):
                row = ''
                for y, node in enumerate(column):
                    print_string = ' ' * maximum_cell_width
                    if node == 0:
                        current_island = island_info[(x,y)]
                        length_curr_island_string = len(str(current_island))
                        # The amount of padding needed for the current cell
                        amount_padding_needed = maximum_cell_width - length_curr_island_string
                        left_padding = ' ' * int(amount_padding_needed / 2)
                        right_padding = ' ' * ceil(amount_padding_needed / 2)
                        print_string = f"{left_padding}{current_island}{right_padding}"
                    row += f" [{print_string}]"
                print(row)
            print()
            print("*======================================================================*")
            print()
        else:
            print("I can't print >3 dimensions...")


    def get_adjacent_positions(self, pos, shape):
        """
        Generate all valid adjacent positions for a given node position.
        """
        shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * len(pos))).T.reshape(-1, len(pos))
        shifts = [tuple(shift) for shift in shifts if not all(s == 0 for s in shift)]  # Exclude no movement
        
        # List to store valid adjacent positions
        adjacent = []
        
        for shift in shifts:
            new_pos = tuple(np.array(pos) + np.array(shift))
            # Check if the new position is within bounds
            if all(0 <= new_pos[dim] < shape[dim] for dim in range(len(pos))):
                adjacent.append(new_pos)
        
        return adjacent


    def is_within_bounds(self, pos):
        """
        Check if the given position is within the bounds of the grid.
        
        Parameters:
        - pos (tuple): The position to check.
        
        Returns:
        - bool: True if the position is within bounds, False otherwise.
        """

        return all(0 <= pos[i] < self.ocean.shape[i] for i in range(len(pos)))


    def island_counting_helper(self, found_nodes, current_node_coords):
        """
        Recursively traverse nested arrays of ints, find all connecting adjacent nodes that have a value of 0 (ie 
        "land") and record their coordinates. Any integer value above 0 within the nested arrays is considered "water".
        Returns an updated set of coordinates that have been scanned as well as a dictionary (coord : island association).

        Arguments:
            ocean: Nested arrays of integers
                Example: ([ [0, 1, 0, 0, 0, 1], ... ])

            found_nodes: Set of node coordinates already scanned within the recursive traversal.
                Examples: { (1,0), (3, 5), ... }
            
            current_node_coords: The tuple coordinates of the node currently being scanned within the recursive traversal
        """
        if found_nodes is None:
            found_nodes = set()

        if not self.is_within_bounds(current_node_coords):
            return found_nodes  # Out of bounds

        value = self.ocean
        for index in current_node_coords:
            value = value[index]
        if value > 0:
            return found_nodes

        if current_node_coords not in found_nodes:
            # Add the current node to the list of found nodes
            found_nodes.add(current_node_coords)
        else:
            return found_nodes
        
        for adj_pos in self.get_adjacent_positions(current_node_coords):
            self.island_counting_helper(found_nodes, adj_pos)
        
        return found_nodes
    

    def count_distinct_groups(self, should_print=False):
        """
        Count distinct groupings of adjacent nodes in an n-dimensional array.
        """
        visited = set()
        groups = 0
        shape = self.ocean.shape
        island_info = dict()

        def bfs(start):
            """
            Perform BFS to find all connected nodes in the same group.
            """
            queue = deque([start])
            visited.add(start)
            
            while queue:
                current = queue.popleft()
                # Explore all adjacent nodes
                for neighbor in self.get_adjacent_positions(current, shape):
                    if neighbor not in visited and self.ocean[neighbor] == self.ocean[start]:
                        island_info[neighbor] = groups
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # Iterate over all positions in the array
        for index, value in np.ndenumerate(self.ocean):
            if index not in visited and value != 1:
                # Found a new group, so perform BFS/DFS to explore it
                groups += 1
                island_info[index] = groups
                bfs(index)
        
        # Fancy printing shtuff
        if should_print:
            print() 
            print(f"NUMBER OF ISLANDS: {groups}!")
            print(groups)
            print()
            self.print_distinct_islands(island_info, groups)
        return groups


    def count_distinct_islands(self, should_print=False):
        """
        Take an ocean (ie nested arrays of ints), and print and count the number all distinct islands. An island is
        defined as a continuous string of adjacent nodes containing the value: 0

        Arguments:
            ocean: Nested arrays of integers
                Example: ([ [0, 1, 0, 0, 0, 1], ... ])
        """
        total_found_nodes = set()
        num_islands = 0
        island_info = dict()
        # Iterate over the ocean, looking for an island
        shape = self.ocean.shape
        # flatten the ocean
        flattened_ocean = self.ocean.flatten()
        # associate real nth dimensional coords for each index of the flattened array
        coords = [np.unravel_index(index, shape) for index in range(flattened_ocean.size)]
        for index, coord in enumerate(coords):
            # if we haven't marked the node yet
            if flattened_ocean[index] < 1 and coord not in total_found_nodes:
                # we've now officially found a new island
                num_islands += 1
                # recursively seek out and mark adjacent nodes
                current_found_nodes = self.island_counting_helper(found_nodes=set(), current_node_coords=coord)
                for found_node in current_found_nodes:
                    # Save the island's info for printing purposes
                    island_info[found_node] = num_islands

                # Add the set of scanned nodes to the running overall list of scanned nodes
                total_found_nodes = total_found_nodes.union(current_found_nodes)

        # Fancy printing shtuff
        if should_print:
            print() 
            print(f"NUMBER OF ISLANDS: {num_islands}!")
            print(num_islands)
            print()
            self.print_distinct_islands(island_info, num_islands)
        return num_islands


    def create_ocean(self, dimension_sizes=None, frequency=None, print=False):
        """
        Initialize a random ocean of "islands" (ie a nested list of ints), an island is defined by a particular (x,y)
        coordinate having the value of "0".

        Example:
        *=================================================================*
            [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ]
            [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ]
            [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ]
            [ 1 ] [ 0 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ] [ 0 ] [ 1 ] [ 0 ]
            [ 0 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ]
            [ 0 ] [ 0 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 0 ] [ 1 ]
            [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ]
            [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 0 ]
            [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ]
            [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ]
            [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 0 ] [ 1 ] [ 0 ]
            [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ]
            [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 1 ]
            [ 0 ] [ 1 ] [ 1 ] [ 1 ] [ 1 ] [ 0 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ]
            [ 1 ] [ 1 ] [ 0 ] [ 0 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ] [ 0 ] [ 1 ]
        *=================================================================*
        """
        try:
            if not dimension_sizes:
                dimension_sizes = self.dimension_sizes
            else:
                self.dimension_sizes = dimension_sizes
            if not frequency:
                frequency = self.frequency
            else:
                self.frequency = frequency
        except AttributeError as exc:
            raise Exception("Provide a length, width, and frequency if the ocean does not already exist.", exc)
        self.ocean = create_nested_numpy_array(
            sizes=self.dimension_sizes,
            frequency=self.frequency,
        )
        if print:
            self.print_distinct_islands()
            # self.print_ocean(self.ocean)
        return self.ocean

    def run_conways_game_of_life_3d(self, iterations=1):
        """
        Run Conway's Game of Life on a 3D grid for a given number of iterations.
        
        Parameters:
        grid (numpy.ndarray): A 3D NumPy array of 1's and 0's, where 1 represents life and 0 represents death.
        iterations (int): Number of iterations to run the game.
        
        Returns:
        numpy.ndarray: The grid after the specified number of iterations.
        """
        # Define the 3D neighborhood kernel (3x3x3 cube) to calculate neighbors
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0  # The center cell itself is not a neighbor

        # Flip the ocean because of a cruel joke I'm seemingly playing on myself
        self.ocean = np.where((self.ocean==0)|(self.ocean==1), self.ocean^1, self.ocean)
        for _ in range(iterations):

            # Fuck you Morgan
            # Count the number of live neighbors for each cell using convolution
            live_neighbors = convolve(self.ocean, kernel, mode='constant', cval=0)

            # Apply the rules of Conway's Game of Life
            new_grid = np.zeros_like(self.ocean)

            # Rule 1 and 3: Live cell with 2 or 3 neighbors survives, else it dies
            new_grid[(self.ocean == 1) & ((live_neighbors == 5) | (live_neighbors == 4) | (live_neighbors == 6) | (live_neighbors == 7) | (live_neighbors == 8))] = 1

            # Rule 2: Dead cell with exactly 3 neighbors becomes a live cell
            new_grid[(self.ocean == 1) & ((live_neighbors == 3) | (live_neighbors == 4) | (live_neighbors == 5) |  (live_neighbors == 6))] = 1

            # Debugging: print number of live cells at each step to monitor population changes
            live_count = np.sum(new_grid)
            print(f"Iteration {_ + 1}, Live cells: {live_count}")


            # Update the grid for the next iteration
            self.ocean = new_grid
        
        self.ocean = np.where((self.ocean==0)|(self.ocean==1), self.ocean^1, self.ocean)
        return self.ocean



def island_count_expectation_value_1d(width=10, frequency=.5):
    return (width*frequency) - ((width-1)*frequency**2)


def island_count_expectation_value_2d(width=2, height=2, frequency=.5):
    """
    this is not correct...
    """
    return (4*frequency) - (6*frequency**2) + (4*frequency**3) - frequency**4


ocean = Ocean(dimension_sizes=[5, 5, 10], frequency=.1)
# ocean.run_conways_game_of_life_3d(iterations=2)
# ocean.ocean = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]]])
ocean.count_distinct_groups(should_print=True)
