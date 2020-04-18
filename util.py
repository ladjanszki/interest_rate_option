class Node:
    '''
    This class represents the nodes in the interest rate tree
    '''
    def __init__(self):
        self.children = [] #This might be a list of tuples
        self.short_rate = 0.0 #Short rate in this world state
        self.payoff = 0.0 #Payoff for immediate exercise
        self.cond_exp = 0.0 #Conditional expectation of the child
        self.max = 0.0 #Maximum of the cond. expectation and the payoff

    def  __str__(self):
        return 'Node object'


class Tree:
    '''
    This class represents the interest rate tree
    '''
    def __init__(self, R0, deltaR, nLevel):

        # Dictionary storing the nodes
        self.nodes = {} 
        for level in range(nLevel):
            self.nodes[level] = {}
            for rateLevel in range(-level, level + 1):
                self.nodes[level][rateLevel] = Node()
      
 
