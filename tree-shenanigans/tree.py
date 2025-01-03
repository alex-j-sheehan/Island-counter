class Node():
    """
    Non-binary tree node class for building out a dependency tree of objects to create with customizations.
    """
    def __init__(self, data):
        self.data = data
        self.children = []
        self.customizations = {}

    def set_single_customization(self, field, value):
        """
        Set a single customization value to the current node, overrides existing values under the same key.
        """
        self.customizations[field] = value

    def add_child(self, obj):
        """
        Add a child to the current node
        """
        self.children.append(obj)

    def add_children(self, children):
        """
        Add multiple children to the node
        """
        for child in children:
            self.children.append(child)

    def find_value(self, value):
        """
        Find a value in the tree
        """
        if self.data == value:
            return self
        else:
            for child in self.children:
                found = child.find_value(value)
                if found:
                    return found
            return None

    def print_nodes_per_level(self):
        nodes_per_level = self.nodes_per_level_helper()
        for key, values in nodes_per_level.items():
            line_string = f"{key}"
            for value in values:
                line_string += f"   {value}"
            print(line_string)

    def nodes_per_level_helper(self, level=1, nodes_per_level=dict()):
        if nodes_per_level.get(level):
            nodes_per_level[level].append(self.data)
        else:
            nodes_per_level[level] = [self.data]
        
        for child in self.children:
            nodes_per_level = child.nodes_per_level_helper(level + 1, nodes_per_level)
        
        return nodes_per_level

    def __str__(self, level=0):
        """
        Overridden str method to allow for proper tree printing
        """
        body = f"fields: {self.customizations}"
        ret = ("\t" * level) + f"{repr(self.data)} {body}" + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        """
        Overridden repr
        """
        return f'<Tree Node {self.data}>'

parent = Node("parent")
child_1 = Node(1)
child_2 = Node(2)
child_3 = Node(3)
child_4 = Node(4)
child_5 = Node(5)
child_6 = Node(6)
child_7 = Node(7)

parent.add_child(child_1) 
parent.add_child(child_2) 
child_1.add_child(child_3)
child_1.add_child(child_4)
child_2.add_child(child_5)

parent.print_nodes_per_level()
