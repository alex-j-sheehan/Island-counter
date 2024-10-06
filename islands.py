import os

from random import randint, random
from math import ceil
import numpy as np
import plotly.graph_objects as go


def create_cube_at_point(x, y, z, size=1):
    # Define the 8 vertices of the cube
    vertices = np.array([
        [x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z],
        [x, y, z + size], [x + size, y, z + size], [x + size, y + size, z + size], [x, y + size, z + size]
    ])

    # Define the 12 triangles composing the cube (2 triangles per face)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [1, 2, 6], [1, 6, 5],  # Right face
        [0, 3, 7], [0, 7, 4]  # Left face
    ])

    return vertices, faces


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
        if len(self.dimension_sizes == 3):
            x, y, z = np.mgrid[:self.ocean.shape[0], :self.ocean.shape[1], :self.ocean.shape[2]]
            x_flat = x.flatten()
            y_flat = y.flatten()
            z_flat = z.flatten()
            values_flat = self.ocean.flatten()

            fig = go.Figure(data=go.Volume(
                x=x_flat, y=y_flat, z=z_flat, value=values_flat,
                isomin=values_flat.min(),
                isomax=values_flat.max(),
                opacity=0.1,  # Adjust for transparency
                surface_count=20,  # Number of isosurfaces
                colorscale='Viridis',  # Color scheme
            ))
            fig.show()
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

            num_islands: Integer, total number of islands found within the ocean. Used for printing padding.
        """
        if len(self.dimension_sizes) == 3:
            values_flat = self.ocean.flatten()

            # Dictionary to store the mapping of int to hex
            int_to_hex_map = {}

            # Example array of integers (from 1 to 10,000)
            for num in island_info.values():
                # If the number hasn't been assigned a hex value, generate one
                if num not in int_to_hex_map:
                    int_to_hex_map[num] = generate_random_hex()

            # Create lists to hold the cube vertices and face indices
            vertices_list = []
            faces_list = []
            colors = []

            vertex_offset = 0  # To keep track of the offset in indices when adding multiple cubes

            for indx, val in enumerate(values_flat):
                coords = np.unravel_index(indx, self.ocean.shape)
                if val == 0:
                    island_id = island_info.get(coords)
                    color = int_to_hex_map.get(island_id, 'magenta')
                    cube_vertices, cube_faces = create_cube_at_point(*coords)

                    # Add the vertices and faces for each cube
                    vertices_list.extend(cube_vertices)
                    faces_list.extend(cube_faces + vertex_offset)
                    colors.extend([color] * len(cube_faces))  # Add color for each face

                    vertex_offset += 8  # Each cube has 8 vertices

            # Convert to numpy arrays for mesh3d
            vertices_list = np.array(vertices_list)
            faces_list = np.array(faces_list)

            # Extract x, y, z coordinates from vertices
            x_vertices = vertices_list[:, 0]
            y_vertices = vertices_list[:, 1]
            z_vertices = vertices_list[:, 2]

            # Create the mesh3d plot
            fig = go.Figure(data=go.Mesh3d(
                x=x_vertices,
                y=y_vertices,
                z=z_vertices,
                i=faces_list[:, 0],
                j=faces_list[:, 1],
                k=faces_list[:, 2],
                facecolor=colors,  # Color for each face
                opacity=0.9,
            ))
            fig.show()
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
                        current_island = island_info[(x, y)]
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

    def get_adjacent_positions(self, pos):
        """
        Generate all possible adjacent positions in N-dimensional space, including diagonals.

        Parameters:
        - pos (tuple): Current position in the grid.

        Returns:
        - list of tuples: List of all adjacent positions, including diagonals.
        """
        # Generate all possible changes to each dimension (-1, 0, or +1)
        shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * len(pos))).T.reshape(-1, len(pos))
        shifts = [tuple(shift) for shift in shifts if not all(s == 0 for s in shift)]  # Exclude no movement
        return [tuple(np.array(pos) + np.array(shift)) for shift in shifts]

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
            self.print_ocean(self.ocean)
        return self.ocean


def island_count_expectation_value_1d(width=10, frequency=.5):
    return (width * frequency) - ((width - 1) * frequency ** 2)


def island_count_expectation_value_2d(width=2, height=2, frequency=.5):
    """
    this is not correct...
    sorry
    """
    return (4 * frequency) - (6 * frequency ** 2) + (4 * frequency ** 3) - frequency ** 4


ocean = Ocean(dimension_sizes=[20, 20, 20], frequency=.02)
ocean.count_distinct_islands(should_print=True)