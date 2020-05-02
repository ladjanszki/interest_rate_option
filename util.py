import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

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
        self.branching = 0 # Branching type of this node
        self.time = time #Timestamp of the given node
        self.rateLevel = rateLevel

        self.RStar = 0 #Short rate in this world state
        self.Q = 0 #Arrow debreu price in this world state
        self.alpha = 0  #Shift to match the zero coupon curve

        self.R = 0

        #self.payoff = 0.0 #Payoff for immediate exercise
        #self.condExp = 0.0 #Conditional expectation of the child
        #self.max = 0.0 #Maximum of the cond. expectation and the payoff


    def  __str__(self):
        strRepr = ''
        strRepr += 'T = {0}\n'.format(self.time) 
        strRepr += 'j = {0}\n'.format(self.rateLevel) 
        strRepr += 'R = {0:8.4f}%\n'.format(self.R * 100) # Rate printed in precentage
        strRepr += 'Q = {0:8.4f}\n'.format(self.Q) 
        #strRepr += str(self.parents)

        return strRepr


class Tree:
    '''
    This class represents the interest rate tree
    This is not a tree actually it is an acyclic directed graph
    '''
    def __init__(self, T, dt, k, theta, sigma, zeroCoupon):

        # Variables
        self.T = T # Maturity
        self.dt = dt #Time step in the tree
        self.nLevel = int(T / dt) + 1 # How many levels the tree has

        self.k = k # Long term rate
        self.theta = theta # Theta from Vasiscek model
        self.sigma = sigma # Volatility

        #self.r0 = None # Starting interest rate
        #self.K = None # Strike price of the interest rate option

        self.zc = zeroCoupon # Observer zero coupon curve to fit the tree to
        self.alphas = np.zeros(self.nLevel, dtype=float) # Shifts per level

        #print(self.zc)

        self.dR = sigma * np.sqrt(3 * dt)

        # Magic numbers for visualization
        self.xSpacing = 4
        self.ySpacing = 150

        # Transition probability dictionary
        self.transProb = {}

        # Creating the nodes
        # TODO: can be factored out to a method
        self.nodes = {} 
        for level in range(self.nLevel):
            self.nodes[level] = {}

            for rateLevel in range(-level, level + 1):
                nodeName = 'node_' + str(level) + '_' + str(rateLevel).replace('-', 'm')
                self.nodes[level][rateLevel] = Node(nodeName, self.dt * level, rateLevel)

        # Setting the root node
        # TODO: can be factored out to a method
        self.nodes[0][0].isRoot = True

        # Building the transition probability dictionary
        self.transProbDict()

        # Adding connections, building the tree
        self.addConnections()

        # Delete orphan nodes
        self.deleteOrphans()

        # Adding the short rates to the nodes
        self.addRates()

    def transProbDict(self):
        '''
        This function builds a dictionary with the transition probabilities 
        for different branching schemes

        The dictionary then save in the tree class and used when the connections in the tree are built

        The probabilities have to be calculated for all rateLevels of the tree (j dependence)
        This can be achieved by calculating for the last level where all rateLevels are present
        '''

        # Short notations for readable expressions
        a = self.theta

        # Building the empty dictionary
        level = self.nLevel - 1 # Building for the last level
        for rateLevel in range(-level, level + 1):
            self.transProb[rateLevel] = {}
            for idx1 in range(-1, 2):
                self.transProb[rateLevel][idx1] = {}
                for idx2 in range(-1, 2): 
                    self.transProb[rateLevel][idx1][idx2] = None

        # Calculating the probabilities
        level = self.nLevel - 1 # Building for the last level
        for rateLevel in range(-level, level + 1):

            # Short notations for readable expressions
            j = rateLevel
            dt = self.dt

            # Branching -1 (down, c)
            self.transProb[j][-1][1]  = (7 / 6) + (1 / 2) * (a**2 * j**2 * dt**2 - 3 * a * j * dt)  
            self.transProb[j][-1][0]  = (-1 / 3) - a**2 * j**2 * dt**2 + 2 * a * j * dt
            self.transProb[j][-1][-1] = (1 / 6) + (1 / 2) * (a**2 * j**2 * dt**2 - a * j * dt)  
 
            # Branching 0 (mid, a)
            self.transProb[j][0][1]  = (1 / 6) + (1 / 2) * (a**2 * j**2 * dt**2 - a * j * dt)  
            self.transProb[j][0][0]  = (2 / 3) - a**2 * j**2 * dt**2
            self.transProb[j][0][-1] = (1 / 6) + (1 / 2) * (a**2 * j**2 * dt**2 + a * j * dt)  
            
            # Branching 1 (up, b)
            self.transProb[j][1][1]  = (1 / 6) + (1 / 2) * (a**2 * j**2 * dt**2 + a * j * dt)  
            self.transProb[j][1][0]  = (-1 / 3) - a**2 * j**2 * dt**2 - 2 * a * j * dt
            self.transProb[j][1][-1] = (7 / 6) + (1 / 2) * (a**2 * j**2 * dt**2 + 3 * a * j * dt)  


        # Chacking if probabilities sum to 1
        level = self.nLevel - 1 
        for rateLevel in range(-level, level + 1):
            for idx1 in range(-1, 2):

                probSum = 0
                for idx2 in range(-1, 2): 
                    probSum += self.transProb[rateLevel][idx1][idx2]

                if np.abs(probSum - 1) >= 10E-8:
                    print('Error in probabilities: ', rateLevel, idx1, probSum)
         
    def addConnections(self):
        '''
        Special branching can be added based on the rateLevel
        No children added for the leaf nodes (last level)

        This function also adds the transition probabilities for each clidren
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
         
                # Adding children list and transition probabilities
                for offset in range(-1, 2):
                    actTransProb = self.transProb[rateLevel][actNode.branching][offset]
                    childTuple = (level + 1, rateLevel + offset + actNode.branching, actTransProb)
                    actNode.children.append(childTuple)

        # Building the parent list
        for level in range(self.nLevel - 1):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                for child in actNode.children:
                    # child[2] is the transition probability from parent to children
                    self.nodes[child[0]][child[1]].parents.append((level, rateLevel, child[2]))

    def addRates(self):
        '''
        Adding the short rate 
        '''

        #Adding the short rate according to the first stage (zero tree mean)
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                # Only adding the rates if not deleted (orphan node)
                if actNode != None:
                    actNode.RStar = rateLevel * self.dR 

        # Second stage
        # TODO: debug this to Hull's example

        # Q and alpha level 0
        self.nodes[0][0].Q = 1
        self.alphas[0] = self.zc[1] / self.dt

        # Q and alpha level 1
        #level = 1

        #for level in range(self.nLevel - 1):
        for level in [1, 2, 3]:
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]
                if actNode != None:
                    actQ = 0
                    for parent in actNode.parents:

                        #print(level, rateLevel)
                        #print(parent)
                        actParent = self.nodes[parent[0]][parent[1]]

                        #print(actParent.Q, parent[2], np.exp(- (self.alphas[level - 1] + parent[1] * self.dR) * self.dt))
                        #print(self.alphas[level - 1], parent[1], self.dR, self.dt)
                        actQ += actParent.Q * parent[2] * np.exp(- (self.alphas[level - 1] + parent[1] * self.dR) * self.dt) 

                    #print('')

                    #self.nodes[level][rateLevel].Q = actQ
                    actNode.Q = actQ



            #print('compute alpha')
            tmp = 0
            # TODO: Revert this loop
            # TODO: Rename tmp to sometring descriptive
            # TODO: After reverting, merge with loop above
            #for rateLevel in range(-level, level + 1):
            for rateLevel in range(level, -level - 1, -1):
                actNode = self.nodes[level][rateLevel]
                if actNode != None:
                    #tmp += self.nodes[level][rateLevel].Q * np.exp(- rateLevel * self.dR * self.dt)
                    tmp += actNode.Q * np.exp(- rateLevel * self.dR * self.dt)
                    print(level, rateLevel, self.nodes[level][rateLevel].Q, - rateLevel * self.dR * self.dt)

            #print(tmp)
            self.alphas[level] = (1 / self.dt) * (np.log(tmp) + (level + 1) *  self.zc[level + 1])

      
            
        # Broadcasting alphas
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                # Only adding the rates if not deleted (orphan node)
                if actNode != None:
                    self.nodes[level][rateLevel].alpha = self.alphas[level]
         
        
        #Adding up alphas and rStars
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                if actNode != None:
                    actNode.R = actNode.RStar + actNode.alpha


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
                        actChild.parents.remove((level, rateLevel, child[2]))
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
                    #nodeLine = '{0} [shape=box label="{1}" pos="{2},{3}!"]'.format(actNode.nodeName, actNode, level * self.xSpacing, rateLevel * self.ySpacing)
                    nodeLine = '{0} [shape=box label="{1}" pos="{2},{3}!"]'.format(actNode.nodeName, actNode, level * self.xSpacing, actNode.R * self.ySpacing)
                    #print(actLine)
                    nodes.append(nodeLine)

                    # Adding the edges to the dot file
                    for child in actNode.children:
 
                        # Children is stored as a tuple of (level, rateLevel) 
                        actChildren = self.nodes[child[0]][child[1]]
                        transProbLabel = child[2] # Transition probability label to all edges
                        edgeLine = '{0} -> {1} [taillabel="{2:8.4f}"]'.format(actNode.nodeName, actChildren.nodeName, transProbLabel) 
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


    #def toPandas(self):
    #    '''
    #    This function puts all the data into a pandas DataFrame to make it easier to debug to Hull's book

    #    '''

    #    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    #    df = pd.DataFrame(columns=['Node'] + labels)

    #    tempDict = dict.fromkeys(['Node'] + labels, None)
    #    tempDict['Node'] = ['R', 'pu', 'pm', 'pd']


    #    labelCounter = 0
    #    for level in range(self.nLevel):
    #        for rateLevel in range(level + 1,-level): # WARNING reverted order to match the Hull book table
    #            actNode = self.nodes[level][rateLevel]

    #            # Adding only existing nodes
    #            if actNode != None:
    #                tempDict[labels[labelCounter]] = actNode.

    #    print(tempDict)

    #    return df
 





 

 
                
              

        
      
 
