# Used to create some dynamic oceans
from random import randint, random
from math import ceil
from itertools import permutations


def print_ocean(ocean):
    """
    Helper method to print out initial, unscanned oceans. Any node with a value of 0 is considered "land"

    Arguments:
        ocean: Nested arrays of integers
            Example: ([ [0, 1, 0, 0, 0, 1], ... ])
    """
    print()
    print("HERE'S THE OCEAN:")
    print("*=============================================================*")
    for column in ocean:
        row = ''
        for node in column:
            row += f" [ {'*' if node == 0 else ' '} ]"
        print(row)
    print("*=============================================================*")
    print()


def print_distinct_islands(island_info, ocean, num_islands):
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
    # Maximum cell width is the number of digets of the last island found (ie island `10` = width of 2)
    maximum_cell_width = len(str(num_islands))
    print()
    print("Results:")
    print("*======================================================================*")
    print()
    for x, column in enumerate(ocean):
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


def island_counting_helper(ocean, found_nodes, current_node_coords):
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
    x = current_node_coords[0]
    y = current_node_coords[1]

    # Scanned coord is "water", no need to proceed so bubble up
    if ocean[x][y] > 0:
        return found_nodes

    # Get the language clear/easier to read
    left = (x - 1, y)
    right = (x + 1, y)
    up = (x, y + 1)
    down = (x, y - 1)
    top_left = (x - 1, y + 1)
    top_right = (x + 1, y + 1)
    bottom_left = (x - 1, y - 1)
    bottom_right = (x + 1, y - 1)
    at_left_edge = x == 0
    at_right_edge = x == (len(ocean) - 1)
    at_top_edge = y == (len(ocean[x]) - 1)
    at_bottom_edge = y == 0

    # Check current node if we've found it already
    if current_node_coords not in found_nodes:
        # Add the current node to the list of found nodes
        found_nodes.add(current_node_coords)

    # Recursively look at the islands adjacent to the current one
    # [up/down, left/right]
    if not at_left_edge and left not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, left)
    if not at_right_edge and right not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, right)
    if not at_top_edge and up not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, up)
    if not at_bottom_edge and down not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, down)
    # [diagonals (top-left, top-right, bottom-left, bottom-right)]
    if not at_top_edge and not at_left_edge and top_left not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, top_left)
    if not at_top_edge and not at_right_edge and top_right not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, top_right)
    if not at_bottom_edge and not at_left_edge and bottom_left not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, bottom_left)
    if not at_bottom_edge and not at_right_edge and bottom_right not in found_nodes:
        found_nodes = island_counting_helper(ocean, found_nodes, bottom_right)

    # done finding the whole island, bubble up!
    return found_nodes


def count_distinct_islands(ocean):
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
    for x, column in enumerate(ocean):
        for y, node in enumerate(column):
            # look at the current node, check if it's an island and if it's already been "found" before
            if node < 1 and (x, y) not in total_found_nodes:
                # Increment the number of distinct islands found
                num_islands += 1
                coords = (x,y)

                # recursively find the whole island
                current_found_nodes = island_counting_helper(ocean, found_nodes=set(), current_node_coords=coords)

                # Save the island associated with the returned nodes
                for found_node in current_found_nodes:
                    island_info[found_node] = num_islands

                # Add the set of scanned nodes to the running overall list of scanned nodes
                total_found_nodes = total_found_nodes.union(current_found_nodes)

    # Fancy printing shtuff
    #print()
    # print(f"NUMBER OF ISLANDS: {num_islands}!")
    #print()
    # print_distinct_islands(island_info, ocean, num_islands)
    return num_islands


def create_ocean(length, width, frequency):
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
    ocean = []
    for _ in range(length):
        nodes = []
        for _ in range(width):
            nodes.append(0 if random() < frequency else 1)
        ocean.append(nodes)
    return ocean


def island_count_expectation_value_1d(width=10, frequency=.5):
    return (width*frequency) - ((width-1)*frequency**2)

def island_count_expectation_value_2d(width=2, height=2, frequency=.5):
    return (4*frequency) - (6*frequency**2) + (4*frequency**3) - frequency**4

# Higher the frequency number => the fewer number of/smaller size of islands
ocean = create_ocean(length=15, width=10, frequency=.5)
print_ocean(ocean)
count_distinct_islands(ocean)


