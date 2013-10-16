class LeafNode(Exception):
    def __init__(self, node_id):
        self.node_id = node_id
    def __str__(self):
        return "Node {} is a leaf".format(self.node_id)
