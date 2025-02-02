import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

import subprocess as sp
 
class Node:
    '''
    This class represents the nodes in the interest rate tree
    '''
    def __init__(self, nodeName, level, rateLevel, time):
        self.nodeName = nodeName
        self.children = [] # Childrens as tuples (level, rateLevel)
        self.parents = [] # Parents of the given node
        self.isRoot = False
        self.centralNodeIndex = 0 # The index of the centran node in the following level
        self.time = time #Timestamp of the given node

        # Positions in the tree
        self.rateLevel = rateLevel
        self.level = level

        self.RStar = 0 #Short rate in this world state
        self.Q = 0 #Arrow debreu price in this world state
        self.alpha = 0  #Shift to match the zero coupon curve

        self.R = 0

        self.payoff = 0.0 #Payoff for immediate exercise
        self.condExp = 0.0 #Conditional expectation of the subtree from this node
        self.max = 0.0 #Maximum of the cond. expectation and the payoff

        self.M = 0 #Conditional expectation of the process value in this node (not the option calculation)

        self.isEarlyExercise = False #If the option can be exercised early in this node
        self.bgColor = '#FFFFFF' # background color for visualization


    def  __str__(self):
        strRepr = ''
        strRepr += 'T = {0}\n'.format(self.time) 
        strRepr += 'j = {0}\n'.format(self.rateLevel) 
        #strRepr += 'br = {0}\n'.format(self.centralNodeIndex) 
        strRepr += 'R = {0:8.4f}%\n'.format(self.R * 100) # Rate printed in precentage
        #strRepr += 'M = {0:8.4f}\n'.format(self.M) 
        strRepr += 'Q = {0:8.4f}\n'.format(self.Q) 
        strRepr += '\u03B1 = {0:8.4f}\n'.format(self.Q) 
        #strRepr += str(self.parents)

        return strRepr


    def isOrphan(self):
        '''
        Boolean function which checks if the node has any parent

        Ture if parent list is []
        False otherwise
        '''

        if self.parents == []:
            return True
        else:
            return False


    def shiftNode(self, alpha):
        '''
        Method to shift the node with a predefined value
        '''

        self.alpha = alpha
        self.R = self.RStar + self.alpha


class Tree:
    '''
    This class represents the interest rate tree
    This is not a tree actually it is an acyclic directed graph
    '''
    def __init__(self, T, dt, k, theta, sigma, zeroCoupon, asCourseMaterial):

        # Basic 
        self.T = T # Maturity
        self.nLevel = int(T / dt) + 1 # How many levels the tree has

        self.earlyExerciseColor = '#BDBDBD' 
        self.noExerciseColor = '#FFFFFF' 

        # Calculation method
        # The tree supports two calculation methods 
        # 1) from the book by Bringo and Mercurio
        # They trim the tree and the conditional expectation do  not have an explicit long term rate dependence
        # 2) from the course material
        # Here k is not calculated and the conditional expectation has an explicit long term rate dependence
        self.asCourseMaterial = asCourseMaterial

        # Process
        self.k = k # Long term rate
        self.theta = theta # Theta from Vasiscek model
        self.sigma = sigma # Volatility

        # Different parametrisations
        self.a = self.theta 
        self.b = self.k 

        # The time step array which currently has equidistant steps
        #self.dt = np.full(self.nLevel - 1, dt, dtype=float)
        self.dt = dt

        # Calculating the variance per level
        self.V2 = (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * self.dt))
        self.V = np.sqrt(self.V2)

        # The interest rate displacement
        self.dR = self.V * np.sqrt(3)
       

        self.zc = zeroCoupon # Observable zero coupon curve to fit the tree to
        self.alphas = np.zeros(self.nLevel, dtype=float) # Shifts per level

        # Magic numbers for visualization
        self.xSpacing = 4
        self.ySpacing = 150


        # Creating the nodes
        self.nodes = {} 
        self.createNodes()

        # Adding the short rates to each node
        self.calcRates()

        # Calculating the process conditional expectations (M)
        self.calcM()

        # Calculating the central children for each node (k)
        self.calcCentralNode()

        # Adding connections and transition probabilities
        self.addConnections()

        # Delete orphan nodes
        self.deleteOrphans()

        # Second stage, fitting the tree to the observed yield curve
        self.shiftTree()

    def setEarlyExercise(self, tAlpha, tBeta):
        '''
        This method sets the isEarlyExercise tag in the Node class for every node
    
        For european options no early exercise is allowed
        For american options early exercise is allowed ina all nodes
        For bermuda optison early exercise is allowed for a given period of time
        '''

        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                if actNode != None:
                    if (tAlpha <= actNode.time) and (actNode.time <= tBeta):
                        actNode.isEarlyExercise = True
                        actNode.bgColor = self.earlyExerciseColor
                    else:
                        actNode.isEarlyExercise = False
                        actNode.bgColor = self.noExerciseColor

    def addConnections(self):
        '''
        Special branching can be added based on the rateLevel
        No children added for the leaf nodes (last level)

        This function also adds the transition probabilities for each clidren

        The children will be added as reference for cleaner code 
        (child, transition probability) tuple will be added
        '''

        for level in range(self.nLevel - 1):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                if actNode != None:
                    probSum = 0
                    centralNode = self.nodes[level + 1][actNode.centralNodeIndex]
                    # Building children list and transition probabilities
                    for offset in range(-1, 2):

                        actChild = self.nodes[level + 1][actNode.centralNodeIndex + offset]
                        #transProb = float(1 / 3) 
                        transProb = self.calcTransProb(actNode, centralNode, offset)
                        probSum += transProb
                        actNode.children.append((actChild, transProb))

                # Checking if probabilities add up to one
                if np.abs(probSum - 1) >= 10E-8:
                    print('Error in probabilities: ', level, rateLevel, probSum)
                    exit(1)

        # Building the parent list (the last level is parent to no other node)
        for level in range(self.nLevel - 1):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                for child, transProb in actNode.children:
                    # child[2] is the transition probability from parent to children
                    #self.nodes[child[0]][child[1]].parents.append((level, rateLevel, child[2]))
                    child.parents.append((actNode, transProb))


    def calcCentralNode(self):
        '''
        The tree to correctly express mean reversion the central node for the branching process 
        have to be calculated as the central node will be as close to the conditional expectation (M)
        as it is possible 
        '''
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                if actNode != None:

                    if self.asCourseMaterial:
                        # No tree structure modification
                        actNode.centralNodeIndex = rateLevel

                    else:
                        # Mean reversion by tree strucure
                        actNode.centralNodeIndex = int(round(actNode.M / self.dR))
                
 

    def calcM(self):
        '''
        This function calculates the conditional expectation of the process from this point
        This is not the conditional expectation used when calculating the option price

        We calculate M for all levels but it will not be used in the last level (no children)
        '''
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                if actNode != None:

                    if self.asCourseMaterial:
                        # Longer expression from course
                        actNode.M = actNode.RStar * np.exp(- self.a * self.dt) + (self.b / self.a) * (1 - np.exp(-self.a * self.dt))

                    else:
                        # Short expression from Brigo - Mercurio
                        actNode.M = actNode.RStar * np.exp(- self.a * self.dt)
 


    def createNodes(self):
        '''
        Creating the empty nodes for in the tree
        And setting the root
        '''
        for level in range(self.nLevel):
            self.nodes[level] = {}

            for rateLevel in range(-level, level + 1):
                nodeName = 'node_' + str(level) + '_' + str(rateLevel).replace('-', 'm')
                #self.nodes[level][rateLevel] = Node(nodeName, self.dt * level, rateLevel)
                self.nodes[level][rateLevel] = Node(nodeName, level, rateLevel, level * self.dt)

        # Setting the root node
        self.nodes[0][0].isRoot = True


    def calcRates(self):
        '''
        Calculating the rates for all nodes

        TODO: The calculatoin can be directly done into Node.R
              Node.RStar can be deleted
        '''

        #Adding the short rate according to the first stage (zero tree mean)
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                # Only adding the rates if not deleted (orphan node)
                if actNode != None:
                    actNode.RStar = rateLevel * self.dR 
                    #print(actNode.RStar)

        #Adding up alphas and rStars
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                if actNode != None:
                    actNode.R = actNode.RStar + actNode.alpha

    def shiftTree(self):
        '''
        This tree calculates the second phase of the interest rate tree calculation
        Aafter the initial flat tree is built it have to be fitted to the observed yield curve
        
        This can be done by a recursive calculation of Arrow-Debreu prices and shift parameters for each level
        '''

        # Root node Q and alpha
        self.nodes[0][0].Q = 1
        self.alphas[0] = self.zc[1] / self.dt

        # calculate Q and alpha
        for level in range(1, self.nLevel):

            alphaTmp = 0   # There will be one alpha per level

            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]
                if actNode != None:

                    actQ = 0 #There will be on Q per node
                    for parent, transProb in actNode.parents:
                        actQ += parent.Q * transProb * np.exp(- (self.alphas[level - 1] + parent.rateLevel * self.dR) * self.dt) 

                    #Saving Q
                    actNode.Q = actQ

                    # Calculatoin alpha
                    alphaTmp += actNode.Q * np.exp(-rateLevel * self.dR * self.dt)
                    #print('{0:4d}{1:4d}{2:8.4f}{3:8.4f}'.format(level, rateLevel, actNode.Q, rateLevel * self.dR * self.dt))

            # Save alpha to its own list (one value per level)
            self.alphas[level] = (1 / self.dt) * (np.log(alphaTmp) + (level + 1) *  self.zc[level + 1])

        # Broadcasting alphas (add to every rate level in a given level)
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                # Only adding the rates if not deleted (orphan node)
                if actNode != None:
                    self.nodes[level][rateLevel].shiftNode(self.alphas[level])

    def calcTransProb(self, actNode, actChild, offset):
        '''
        Given a parent and a child this function can calculate the transition probability between then
  
        Offset is the parameter is down (-1) middle(0) or up(1) probability have to be calculated
        '''

        #print(actNode)
        #print(actChild)
        #print(offset)

        eta = actNode.M - actChild.RStar #Transition probabilities should be calculated from the non-shifted tree

        if offset == -1: # Down
            #print('Down')
            return (1 / 6) + (eta**2 / (6 * self.V2)) + (eta / 2 * np.sqrt(3) * self.V)
        elif offset == 0: # Middle
            #print('Mid')
            return (2 / 3) - (eta**2 / (3 * self.V2)) 
        elif offset == 1: #Up
            #print('Up')
            return (1 / 6) + (eta**2 / (6 * self.V2)) - (eta / 2 * np.sqrt(3) * self.V)
        else:
            print('Error: Unknown branching scheme!')
            #exit(1)

        

    
 
    def deleteOrphans(self):
        '''
        Because of special tree building procedure ther can be orphan nodes with no parents
        These should be removed
        ''' 

        # Delete subgraph if node has no parent
        for level in range(self.nLevel):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]

                if (actNode.isOrphan()) and (not actNode.isRoot):
                    for child, transProb in actNode.children:
                        #actChild = self.nodes[child[0]][child[1]]
                        #actChild.parents.remove((level, rateLevel, child[2]))
                        child.parents.remove((actNode, transProb))
        
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
                    nodeLine = '{0} [shape=box label="{1}" pos="{2},{3}!" style=filled fillcolor="{4}" ]'.format(actNode.nodeName, actNode, level * self.xSpacing, actNode.R * self.ySpacing, actNode.bgColor)
                    nodes.append(nodeLine)

                    # Adding the edges to the dot file
                    for child, transProb in actNode.children:
 
                        # Children is stored as reference in a tuple with transition probability
                        edgeLine = '{0} -> {1} [taillabel="{2:8.4f}"]'.format(actNode.nodeName, child.nodeName, transProb) 
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

    def pricing(self, strike):
        '''
        This method prices an option with given strike and early exercise
        The early exercise is alloweb for time  t >= tAlpha and t <= tBeta 

        The method uses the previously built trinomial tree
        '''

        # Calculating payoffs for the last level
        level = self.nLevel - 1
        for rateLevel in range(-level, level + 1):
            actNode = self.nodes[level][rateLevel]
            if actNode != None:
                actNode.payoff = np.maximum(0, actNode.R - strike)
                actNode.condExp = 0.0
                 
                # The maximum will be used in lower levels
                actNode.max = np.maximum(actNode.payoff, actNode.condExp)
        

        # Propagating the values back
        for level in range(self.nLevel -2, -1, -1):
            for rateLevel in range(-level, level + 1):
                actNode = self.nodes[level][rateLevel]
                if actNode != None:
                    actNode.payoff = np.maximum(0, actNode.R - strike)

                    actNode.condExp = 0
                    for child, transProb in actNode.children:
                        #actChild = self.nodes[child[0]][child[1]]
                        # child[2] is the transition probability from parent to actual children
                        #actNode.condExp += child[2] * actChild.max * np.exp(- self.dt * actNode.R)
                        #actNode.condExp += child[2] * actChild.max * np.exp(- self.dt * actNode.R)
                        actNode.condExp += transProb * child.max * np.exp(- self.dt * actNode.R)


                    # If early exercise is possible in this node the maximum of the payoff and cond. expacation is stored
                    # If no early exercise pos the cond. expectation stored
                    if actNode.isEarlyExercise: 
                        actNode.max = np.maximum(actNode.payoff, actNode.condExp)
                    else:
                        actNode.max = actNode.condExp

        # returning the price from the root node
        return self.nodes[0][0].max
                        

