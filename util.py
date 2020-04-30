import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import subprocess as sp
 
class Node:
    '''
    This class represents the nodes in the interest rate tree
    '''
    def __init__(self, nodeName, time, rateLevel):
        self.nodeName = nodeName
        self.children = [] # Childrens as tuples (level, rateLevel)
        self.parents = [] # Parents of the given node
        self.isRoot = False
        self.R = None #Short rate in this world state
        #self.payoff = 0.0 #Payoff for immediate exercise
        #self.condExp = 0.0 #Conditional expectation of the child
        #self.max = 0.0 #Maximum of the cond. expectation and the payoff
        self.branching = 0
        self.time = time #Timestamp of the given node
        self.rateLevel = rateLevel


    def  __str__(self):
        strRepr = ''
        strRepr += 'T = {0}\n'.format(self.time) 
        strRepr += 'j = {0}\n'.format(self.rateLevel) 
        strRepr += 'R = {0:8.4f}\n'.format(self.R) 
        #strRepr += str(self.parents)

        return strRepr


class Tree:
    '''
    This class represents the interest rate tree
    This is not a tree actually it is an acyclic directed graph
    '''
    def __init__(self, T, dt, k, theta, sigma):

        # Variables
        self.T = T # Maturity
        self.dt = dt #Time step in the tree
        self.nLevel = int(T / dt)# How many levels the tree has

        self.k = k # Long term rate
        self.theta = theta # Theta from Vasiscek model
        self.sigma = sigma # Volatility

        #self.r0 = None # Starting interest rate
        #self.K = None # Strike price of the interest rate option

        self.dR = sigma * np.sqrt(3 * dt)

        # Magic numbers for visualization
        self.xSpacing = 4
        self.ySpacing = 2

        # Creating the nodes
        self.nodes = {} 
        for level in range(self.nLevel):
            self.nodes[level] = {}

            for rateLevel in range(-level, level + 1):
                nodeName = 'node_' + str(level) + '_' + str(rateLevel).replace('-', 'm')
                self.nodes[level][rateLevel] = Node(nodeName, self.dt * level, rateLevel)

        # Setting the root node
        self.nodes[0][0].isRoot = True

        # Adding connections, building the tree
        self.addConnections()

        # Adding the short rates to the nodes
        self.addRates()

        # Delete orphan nodes
        self.deleteOrphans()

 
    def addConnections(self):
        '''
        Special branching can be added based on the rateLevel
        No children added for the leaf nodes (last level)
        '''

        # Calculating j_max and j_min from process parameters
        jMax = int(np.ceil(0.184 / (self.theta * self.dt))) 
        jMin = -jMax

        for level in range(self.nLevel - 1):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                # Setting special branching 
                if rateLevel >= jMax:
                    actNode.branching = -1
                if rateLevel <= jMin:
                    actNode.branching = +1
         
                
                for offset in range(-1, 2):
                    actNode.children.append((level + 1, rateLevel + offset + actNode.branching))

        # Building the parent list
        for level in range(self.nLevel - 1):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                for child in actNode.children:
                    self.nodes[child[0]][child[1]].parents.append((level, rateLevel))

    def addRates(self):
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]
                actNode.R = rateLevel * self.dR 


    def deleteOrphans(self):
        '''
        Because of special tree building procedure ther can be orphan nodes with no parents
        These should be removed
        ''' 

        # Delete subgraph if node has no parent
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]
                if (actNode.parents == []) and (not actNode.isRoot):
                    for child in actNode.children:
                        actChild = self.nodes[child[0]][child[1]]
                        actChild.parents.remove((level, rateLevel))
                    self.nodes[level][rateLevel] = None
 

    def toGraphviz(self, dotFileName):
        '''
        This function generates a graphviz dot file from the current tree
        This dot file can be used to visualize the tree 
        It is very helpful for debugging
        '''

        nodes = []
        edges = []

        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                # Adding only existing nodes
                if actNode != None:

                    # Adding the node to the dot file
                    nodeLine = '{0} [shape=box label="{1}" pos="{2},{3}!"]'.format(actNode.nodeName, actNode, level * self.xSpacing, rateLevel * self.ySpacing)
                    #print(actLine)
                    nodes.append(nodeLine)

                    # Adding the edges to the dot file
                    for child in actNode.children:
 
                        # Children is stored as a tuple of (level, rateLevel) 
                        actChildren = self.nodes[child[0]][child[1]]
                        edgeLine = '{0} -> {1}'.format(actNode.nodeName, actChildren.nodeName) 
                        edges.append(edgeLine)
                

        # Writing into dot file
        with open(dotFileName, 'w+') as dotFile:
            dotFile.write('digraph{\n')

            # Lines for the nodes
            dotFile.write('\n')
            dotFile.write('\n'.join(nodes))
            dotFile.write('\n')

            # Lines for the edges
            dotFile.write('\n')
            dotFile.write('\n'.join(edges))
            dotFile.write('\n')

            dotFile.write('}\n')
            dotFile.write('\n')


    def visualize(self, outFormat):

        baseName = 'out'
        dotFileName = baseName  + '.dot'

        # Write self representation into a dot file
        self.toGraphviz(dotFileName)

        if outFormat == 'png':
            pngFileName = baseName + '.png'
            
            # Call graphviz
            command = ['neato', '-Tpng', dotFileName, '-o',  pngFileName]
            result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
            
            # Show image generated by graphviz
            img = mpimg.imread(pngFileName)
            imgplot = plt.imshow(img)
            plt.axis('off')
            plt.show()  

        elif outFormat == 'pdf':
            pdfFileName = baseName + '.pdf'
            
            # Call graphviz
            command = ['neato', '-Tpdf', dotFileName, '-o',  pdfFileName]
            result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)

            # Call pdf viewer
            command = ['evince', pdfFileName] 
            result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)

        else:
            print('ERROR: Unknown output format' + str(outFormat))
            exit(1)

# Forward rates
forward = {}
forward[0.0] = 4.5
forward[0.5] = 5.0
forward[1.0] = 5.5
forward[1.5] = 4.5
forward[2.0] = 4.0
forward[2.5] = 4.5
forward[3.0] = 4.8
forward[3.5] = 5.0
forward[4.0] = 5.0
forward[4.5] = 4.4
forward[5.0] = 4.5 
forward[5.5] = 4.8
forward[6.0] = 4.5
forward[6.5] = 4.2
forward[7.0] = 4.3

 


 
                
              

        
      
 
