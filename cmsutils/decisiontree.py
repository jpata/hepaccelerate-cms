import numpy as np
import copy
import sys

class DecisionTreeNode:
    NUMPY_LIB = np
    def __init__(self, varname, cut):
        self.varname = varname
        self.cut = cut
        self.cacheval = None
        
        self.id = None
        self.parent = None
    
    def add_child_left(self, child):
        self.child_left = child
        child.parent = self
        
    def add_child_right(self, child):
        self.child_right = child
        child.parent = self
        
    def predict(self, n, data, ret=None, mask=None):
        if mask is None:
            mask = self.NUMPY_LIB.ones(n, dtype=np.bool)
            
        if ret is None:
            ret = self.NUMPY_LIB.zeros(n)

        my_mask = self.mask_right(data)
        
        self.child_right.predict(n, data, ret=ret, mask=mask & my_mask)
        self.child_left.predict(n, data, ret=ret, mask=mask & self.NUMPY_LIB.invert(my_mask))
        
        return ret
    
    def mask_right(self, data):
        return data[self.varname] > self.cut
    
    def is_leaf(self):
        return (self.child_left is None) and (self.child_right is None)
    
    def __repr__(self):
        return "Node(nodes={0}, leaves={1})".format(
            len(self.get_all_nodes()), len(self.get_all_leaves())
        )
    
    def get_all_leaves(self):
        leaves = []
        leaves += self.child_left.get_all_leaves()
        leaves += self.child_right.get_all_leaves()
        return leaves
    
    def get_all_nodes(self):
        nodes = [self]
        nodes += self.child_left.get_all_nodes()
        nodes += self.child_right.get_all_nodes()
        return nodes
    
    def assign_ids(self):
        for inode, node in enumerate(self.get_all_nodes()):
            node.id = inode
        for ileaf, leaf in enumerate(self.get_all_leaves()):
            leaf.id = ileaf
            leaf.value = ileaf
            
    def make_dot(self):
        import graphviz
        dg = graphviz.Digraph(comment='Categorization')
        DecisionTreeNode.__make_dot_recursive(self, dg)
        return dg
    
    def get_depth(self, d=0):
        if self.parent is None:
            return d
        return self.parent.get_depth(d+1)
    
    @staticmethod
    def __make_dot_recursive(dt, dotgraph):       
        if isinstance(dt, DecisionTreeNode):
            label = "{0}>{1}".format(dt.varname, dt.cut)
            name = "Node({0})".format(dt.id)
            dotgraph.node(name, label=label, shape="rectangle")
            child_name_l = DecisionTreeNode.__make_dot_recursive(dt.child_left, dotgraph)
            child_name_r = DecisionTreeNode.__make_dot_recursive(dt.child_right, dotgraph)
            dotgraph.edge(name, child_name_l, label="n")
            dotgraph.edge(name, child_name_r, label="y")
        elif isinstance(dt, DecisionTreeLeaf):
            name = "Cat({0})".format(dt.value)
            dotgraph.node(name, shape="ellipse")
        return name
    
class DecisionTreeLeaf:
    NUMPY_LIB = np
    def __init__(self):
        self.value = None
        self.id = None
        self.parent = None

    def predict(self, n, data, ret, mask):
        ret[mask] = self.value
        return ret, mask
    
    def __repr__(self):
        s = "Leaf({0})".format(self.value)
        return s
    
    def get_all_leaves(self):
        return [self]
    
    def get_all_nodes(self):
        return []

    def get_depth(self, d=0):
        if self.parent is None:
            return d
        return self.parent.get_depth(d+1)
    
    def assign_ids(self):
            self.id = self.value

def make_random_node(valid_variables, variable_cuts, variable_maxcuts_ind=None):
    random_var = valid_variables[np.random.randint(len(valid_variables))]
    min_idx = 0
    if variable_maxcuts_ind:
        min_idx = variable_maxcuts_ind[random_var]+1
    random_cut = variable_cuts[random_var][np.random.randint(
        min_idx, len(variable_cuts[random_var])
    )]

    new_node = DecisionTreeNode(random_var, random_cut)
    new_node.add_child_left(DecisionTreeLeaf())
    new_node.add_child_right(DecisionTreeLeaf())
    return new_node

def grow_randomly(dt, variable_cuts):
    leaves = [(l, l.get_depth()) for l in dt.get_all_leaves()]
    leaves_lowest = sorted(leaves, key=lambda x: x[1])
    leaves_lowest = [l[0] for l in leaves_lowest][:4]
    ileaf_random = np.random.randint(len(leaves_lowest))
    leaf = leaves_lowest[ileaf_random]
    parent = leaf.parent

    valid_variables = list(variable_cuts.keys())
    variable_maxcuts_ind = {v: 0 for v in valid_variables}
    p = leaf.parent
    while True:
        variable_maxcuts_ind[p.varname] = variable_cuts[p.varname].index(p.cut)
        if variable_maxcuts_ind[p.varname] == len(variable_cuts[p.varname]) - 1:
            valid_variables.pop(valid_variables.index(p.varname))
        p = p.parent
        if p is None:
            break
    
    if len(valid_variables) > 0:
        new_node = make_random_node(valid_variables, variable_cuts, variable_maxcuts_ind)

        if not parent is None:
            if parent.child_left == leaf:
                parent.add_child_left(new_node)
            else:
                parent.add_child_right(new_node)

        dt.assign_ids()

def make_random_tree(varlist, num_iters):
    dtc = make_random_node(list(varlist.keys()), varlist)

    for i in range(num_iters):
        try:
            grow_randomly(dtc, varlist)
        except ValueError as e:
            pass
        dtc.assign_ids()
    return dtc
 
def prune_randomly(dt):
    nodes = dt.get_all_nodes()
    nodes_d2 = [n for n in nodes if n.get_depth()>=1]
    if len(nodes_d2) > 1:
        inode_random = np.random.randint(len(nodes_d2))
        node = nodes_d2[inode_random]

        leaf = DecisionTreeLeaf()
        parent = node.parent
        if node == parent.child_left:
            parent.add_child_left(leaf)
        else:
            parent.add_child_right(leaf)
        dt.assign_ids()

def generate_cut_trees(num_trees, varlist, starting_tree, max_leaves=12):
    all_trees = []
    dtc = copy.deepcopy(starting_tree)

    for i in range(num_trees):

        for j in range(3):        
            if len(dtc.get_all_leaves()) > max_leaves:
                prune_randomly(dtc)
            else:
                try:
                    grow_randomly(dtc, varlist)
                except ValueError as e:
                    pass
 
        all_trees += [copy.deepcopy(dtc)]
    return all_trees
