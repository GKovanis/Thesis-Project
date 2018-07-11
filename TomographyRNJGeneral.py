import numpy as np
from math import log10
from math import sqrt
import time
import networkx as nx
import matplotlib.pyplot as plt
import pydot
import csv


class Graph(object):
    def __init__(self):
        self.root = None #root/source node is the start of the graph/tree and multicast source
        self.nodes = []
        self.leaves = []

class Node(object):
    def __init__(self):
        self.father = None #node's parent
        self.id = None #ID of node
        self.data = [] # tcpdump of the multicast packets of the end-nodes (and only end-nodes, internal nodes do not have data)
        self.children = []


#In Packet Loss, we only care about the packet ID
def GetLinkLossDumps():
    #Tcpdumps of source node stored at root.data
    tcpfile = [line.rstrip('\n') for line in open('dumps/n1.txt')] #opens the tcpdump as a list of strings, we suppose that source connects only to one router to the rest of tree
    for line in range(len(tcpfile)):
        if "tos 0x7" in tcpfile[line]: # tos 0x7 is the characteristic I chose to distinguish my UDP packets used for tomography
            temp = tcpfile[line].split()
            graph.root.data.append(int(temp[7].replace(",",""))) #We keep only the packet ID
    #tcpdump of every leave/destination node stored at node.data
    for i in range(len(graph.leaves)):
        filename = "dumps/n%d.txt" % (graph.leaves[i].id) #example tcpdump file path "thesisdumps/1/n1" if node 1 is a leaf
        tcpfile = [line.rstrip('\n') for line in open(filename)]
        for line in range(len(tcpfile)):
            if "tos 0x7" in tcpfile[line]: # tos 0x7 is the characteristic I chose to distinguish my UDP packets used for tomography
                temp = tcpfile[line].split()
                graph.leaves[i].data.append(int(temp[7].replace(",",""))) #We keep only the packet ID


#In Link Delay and Utilization, we need both the packet ID and the timestamp of the packet
def GetLinkDelayDumps():
    #Tcpdumps of source node stored at root.data
    tcpfile = [line.rstrip('\n') for line in open('dumps/n1.txt')] #opens the tcpdump as a list of strings, we suppose that source connects only to one router to the rest of tree
    for line in range(len(tcpfile)):
        if "tos 0x7" in tcpfile[line]: # tos 0x7 is the characteristic I chose to distinguish my UDP packets used for tomography
            temp = tcpfile[line].split()
            graph.root.data.append([temp[0], int(temp[7].replace(",",""))]) #We keep the timestamp and the packet ID
    #tcpdump of every leave/destination node stored at node.data
    for i in range(len(graph.leaves)):
        filename = "dumps/n%d.txt" % (graph.leaves[i].id) #example tcpdump file path "thesisdumps/1/n1" if node 1 is a leaf
        tcpfile = [line.rstrip('\n') for line in open(filename)]
        for line in range(len(tcpfile)):
            if "tos 0x7" in tcpfile[line]: # tos 0x7 is the characteristic I chose to distinguish my UDP packets used for tomography
                temp = tcpfile[line].split()
                graph.leaves[i].data.append([temp[0], int(temp[7].replace(",",""))]) #We keep the timestamp and the packet ID
    for node in range(len(graph.leaves)):
        TimestampsIntoDelay(graph.root.data,graph.leaves[node].data,node) #we need to turn each timestamp into path delay for each packet
    #root's delay is 0 in all packets (starting point)
    for k in range(len(graph.root.data)):
        graph.root.data[k][0] = float(0)

#Function that measures path Delay from a timestamp, in our algorithm turns every initial leaf's timestamps into delays (difference between start and finish)
def TimestampsIntoDelay(dump1,dump2,node):
    startingpackets=len(dump1) #tcpdump of start node
    endingpackets = len(dump2) #tcpdump of end node
    for packet in range(endingpackets):
        i = 0 # if we are sure that the packets will arive in order, i = packet for faster runtime
        #find packets with same ID
        while (dump1[i][1] != dump2[packet][1]):
            i += 1
        #measure delay for each packet
        #seconds difference
        timestamp1 = dump1[i][0]
        timestamp2 = dump2[packet][0]
        secondsdiff = (int(timestamp2[0:2])*3600+int(timestamp2[3:5])*60+int(timestamp2[6:8]))-(int(timestamp1[0:2])*3600+int(timestamp1[3:5])*60+int(timestamp1[6:8]))
        #fractions of second
        fraction1 = float("0"+timestamp1[8:15])
        fraction2 = float("0"+timestamp2[8:15])
        #delay
        packetdelay=float("{0:.10f}".format(float(secondsdiff)+fraction2-fraction1))
        graph.leaves[node].data[packet][0] = packetdelay #change timestamp with delay

# Function that estimates the distances based on the link loss parameter
def EstimateDistancesLoss():
    # At this point, graph.nodes = U (= source + destination nodes)
    NumberOfNodes = len(graph.nodes)
    # Matrix is symmetric -> We only need to traverse through upper triangular and then complete the symmetrical elements
    # Also, diagonal of the Matrix will be zero (by definition d(i,i) == 0)
    for i in range(NumberOfNodes):
        Xi = len(graph.nodes[i].data)/TotalProbes
        for j in range(i+1,NumberOfNodes):
            # How the distance metric is calculated can be seen in the provided documentation
            Xj = len(graph.nodes[j].data)/TotalProbes
            XiXj = len(set(graph.nodes[i].data)&set(graph.nodes[j].data))/TotalProbes
            distance = log10(Xi*Xj/XiXj**2)
            #Symmetric matrix
            EstDistMatrix[graph.nodes[i].id][graph.nodes[j].id] = distance
            EstDistMatrix[graph.nodes[j].id][graph.nodes[i].id] = distance

# Function that estimates the distances based on the link delay variance parameter
def EstimateDistancesDelayVar():
    # At this point, graph.nodes = U (= source + destination nodes)
    NumberOfNodes = len(graph.nodes)
    # Matrix is symmetric -> We only need to traverse through upper triangular and then complete the symmetrical elements
    # Also, diagonal of the Matrix will be zero (by definition d(i,i) == 0)
    for i in range(NumberOfNodes):
        meanTi = sum([graph.nodes[i].data[k][0] for k in range(len(graph.nodes[i].data))])/len(graph.nodes[i].data)
        for j in range(i+1,NumberOfNodes):
            # How the distance metric is calculated can be seen in the provided documentation
            meanTj = sum([graph.nodes[j].data[k][0] for k in range(len(graph.nodes[j].data))])/len(graph.nodes[j].data)
            # Compute the variances
            varTi = (sum([(graph.nodes[i].data[k][0]-meanTi)**2 for k in range(len(graph.nodes[i].data))]))/(len(graph.nodes[i].data)-1)
            varTj = (sum([(graph.nodes[j].data[k][0]-meanTj)**2 for k in range(len(graph.nodes[j].data))]))/(len(graph.nodes[j].data)-1)
            # Find Common ID between the 2 nodes' packets
            CommonIDs = []
            for k1 in range(len(graph.nodes[i].data)):
                for k2 in range(len(graph.nodes[j].data)):
                    if (graph.nodes[i].data[k1][1] == graph.nodes[j].data[k2][1]):
                        CommonIDs.append(graph.nodes[i].data[k1][1])
            # Compute the covariance
            covTiTj = Covariance(i,j,CommonIDs,meanTi,meanTj)
            distance = varTi + varTj - 2*covTiTj
            # Symmetric matrix
            EstDistMatrix[graph.nodes[i].id][graph.nodes[j].id] = distance
            EstDistMatrix[graph.nodes[j].id][graph.nodes[i].id] = distance

"""
# Function that estimates the distances based on the link utilization parameter
def EstimateDistancesUtil():
    # At this point, graph.nodes = U (= source + destination nodes)
    NumberOfNodes = len(graph.nodes)
    # Epsilon is a small value to acount for possible measurement noise, defined by user
    epsilon = 0.00001
    # Matrix is symmetric -> We only need to traverse through upper triangular and then complete the symmetrical elements
    # Also, diagonal of the Matrix will be zero (by definition d(i,i) == 0)
    for i in range(NumberOfNodes):
        minTi = min([graph.nodes[i].data[k][0] for k in range(len(graph.nodes[i].data))])
        YiPackets = [graph.nodes[i].data[k][1] for k in range(len(graph.nodes[i].data)) if (graph.nodes[i].data[k][0]-minTi <= epsilon)]
        Yi = len(YiPackets)/TotalProbes
        for j in range(i+1,NumberOfNodes):
            # How the distance metric is calculated can be seen in the provided documentation
            minTj = min([graph.nodes[j].data[k][0] for k in range(len(graph.nodes[j].data))])
            YjPackets = [graph.nodes[j].data[k][1] for k in range(len(graph.nodes[j].data)) if (graph.nodes[j].data[k][0]-minTj <= epsilon)]
            Yj = len(YjPackets)/TotalProbes
            YiYj = len(set(YiPackets)&set(YjPackets))/TotalProbes
            distance = log10(Yi*Yj/YiYj**2)

            # Symmetric matrix
            EstDistMatrix[graph.nodes[i].id][graph.nodes[j].id] = distance
            EstDistMatrix[graph.nodes[j].id][graph.nodes[i].id] = distance
"""


# Function that computes the covariance of nodes i,j
def Covariance(i,j,CommonIDs,meanTi,meanTj):
    #Initiliazations
    covar = 0
    pos1 = 0
    pos2 = 0
    length1 = len(graph.nodes[i].data)
    length2 = len(graph.nodes[j].data)
    for packetID in CommonIDs:
        #find position of packetID in node i
        for k1 in range(pos1,length1):
            if (graph.nodes[i].data[k1][1] == packetID):
                pos1=k1
                break
        #find position of packetID in node j
        for k2 in range(pos2,length2):
            if (graph.nodes[j].data[k2][1] == packetID):
                pos2=k2
                break
        covar += (graph.nodes[i].data[pos1][0]-meanTi)*(graph.nodes[j].data[pos2][0]-meanTj)
    covar = covar/(len(CommonIDs)-1)
    return covar


def EstimateScoreFunction():
    # At this point, graph.leaves = D (= destination nodes)
    NumberOfLeaves = len(graph.leaves)
    # Matrix is symmetric -> We only need to traverse through upper triangular and then complete the symmetrical elements
    # Also, diagonal of the Matrix will be equal to zero (we need pair of nodes)
    for i in range(NumberOfLeaves):
        for j in range(i+1,NumberOfLeaves):
            # Score Function is calulated like this:
            # ρ(i,j) = (d(s,i)+d(s,j)-d(i,j))/2
            score = (EstDistMatrix[0][graph.leaves[i].id] + EstDistMatrix[0][graph.leaves[j].id] - EstDistMatrix[graph.leaves[i].id][graph.leaves[j].id])/2
            #Symmetric matrix
            ScoreFunction[graph.leaves[i].id][graph.leaves[j].id] = score
            ScoreFunction[graph.leaves[j].id][graph.leaves[i].id] = score

# Function that calculates Δ(delta) so that the General Tree algorithm can be properly implemented
def CalculateDelta():
    if (param == 'loss'):
        successrate = 0.9995 #should be equal to the minimum link length in terms of loss, changes based on each topology
        delta = -log10(successrate)
    elif (param == 'delayvar'):
        delta = 0.000001 #should be equal to the minimum link length in terms of delay variance, changes based on each topology
    else:
        pass #Link Utilization not measured in our algorithm
    return delta

# Function that visualizes the discovered topology/tree in a .png file
def DrawTopology(param):
    #Create Graph
    G = pydot.Dot(graph_type='graph') #G =nx.Graph()
    for i in range(len(graph.nodes)-1,-1,-1):
        for j in range(len(graph.nodes[i].children)):
            edge = pydot.Edge(graph.nodes[i].id,graph.nodes[i].children[j].id)
            G.add_edge(edge)

    #Draw Graph with desired Parameters
    G.write_png('Results/'+param+'.png')

# Function that writes the results for each inference parameter in a .csv file
def ExtractResults(param):
    # Success/Loss Rate of each Link
    if (param == 'loss'):
        with open('Results/Loss.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Link', 'Success Rate'])
            for i in range(1,len(graph.nodes)):
                SuccessRate = EstDistMatrix[graph.nodes[i].father.id][graph.nodes[i].id]
                SuccessRate = 10**(-SuccessRate)
                filewriter.writerow([graph.nodes[i].id,SuccessRate])
    # Delay Variance of each Link
    elif (param == 'delayvar'):
        with open('Results/DelayVariance.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Link', 'Delay Variance'])
            for i in range(1,len(graph.nodes)):
                #LinkDelayVar = sqrt(EstDistMatrix[graph.nodes[i].father.id][graph.nodes[i].id]) ###If I want the Standard Deviation instead of Variance
                LinkDelayVar = EstDistMatrix[graph.nodes[i].father.id][graph.nodes[i].id]
                filewriter.writerow([graph.nodes[i].id,LinkDelayVar])
    # Utilization of each LinkUtil
    else:
        """
        with open('Results/Utilization.csv', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Link', 'Utilization'])
            for i in range(1,len(graph.nodes)):
                LinkUtil = EstDistMatrix[graph.nodes[i].father.id][graph.nodes[i].id]
                LinkUtil = 10**(-LinkUtil)
                filewriter.writerow([graph.nodes[i].id,LinkUtil])
        """
        pass




### Start of Script ###

# input: Destination Nodes' IDs (Leaves) are given in the DstNodes.txt file
# Create a list with all the Destination Nodes=
DstNodes = [line.rstrip('\n').split(' ') for line in open('DstNodes.txt')]
DstNodes = list(map(int,DstNodes[0]))

# All the inference parameters we want to measure
inferparams = ['loss','delayvar','utilization']

# Perform the algorithm for each inference parameter in the inferparams list
for param in inferparams:

    # Initial Graph Creation
    # V = {s}  : only source node initially on graph
    # E = { }  : no edges created initially

    graph = Graph()

    #creation of source node
    node = Node()
    node.id = 0 #node ID of root is 0
    graph.root = node
    graph.nodes.append(graph.root)

    # Destination Nodes and Graph leaves are the same
    # So we create the graph leaves (without any edges yet) to be able to extract the tcpdumps correctly
    for i in range(len(DstNodes)):
        node = Node()
        node.id = DstNodes[i]
        graph.nodes.append(node)
        graph.leaves.append(node)

    ######### Algorithm: Rooted Neighbor-Joining (RNJ) Algorithm for Binary Trees #########


    #We don't know number of nodes, so we start giving ID numbers to new nodes, starting from max ID of the existing Destination nodes
    FreeID = max(DstNodes) + 1

    #Get the tcpdumps for the root node and the leaves
    if (param == 'loss'):
        GetLinkLossDumps()
    elif (param == 'delayvar'):
        GetLinkDelayDumps()
    else:
        break #delete if you want to measure link utilization too
        pass #GetLinkDelayDumps() used also for utilization

    #Total Probes are equal to the probes sent from the source
    TotalProbes = len(graph.root.data)

    # Estimated Distance Matrix, default size = up to 200 nodes topology
    # Holds the distance metric values for each path,
    # (i,j) element -> Distance metric of path from node i to node j, d(i,j)
    EstDistMatrix = np.zeros((200,200),dtype='f')

    #Create the Estimate Distances Matrix
    if (param == 'loss'):
        EstimateDistancesLoss()
    elif(param == 'delayvar'):
        EstimateDistancesDelayVar()
    else:
        pass #EstimateDistancesUtil() used

    # Step 1

    # Score Function matrix, default size = up to 200 nodes topology (keep same with Estimated Distance matrix)
    # Hold the score function for each pair of nodes i,j
    # (i,j) element -> distance metric for pair of nodes i,j, ρ(i,j)
    ScoreFunction = np.zeros((200,200),dtype='f')

    EstimateScoreFunction()

    # necessary to start the algorithm correctly, normally we shouldn't append destination nodes upon creation but it helped the tcpdumps function
    graph.nodes = []
    graph.nodes.append(graph.root)

    # Step 2.1
    while (len(graph.leaves) != 1):
        # Find i*,j* in D with the largest ScoreFunction (tie is broken arbitrarily as we only take the first occurence)

        NumberOfLeaves = len(graph.leaves)
        # max initialization
        maxScore = 0
        Istar=Jstar = 0
        # find the max score
        for i in range(NumberOfLeaves):
            for j in range(i+1,NumberOfLeaves):
                if (ScoreFunction[graph.leaves[i].id][graph.leaves[j].id] >= maxScore):
                    maxScore = ScoreFunction[graph.leaves[i].id][graph.leaves[j].id]
                    Istar = graph.leaves[i].id
                    Jstar = graph.leaves[j].id


        #Create a node f as parent of i* and j*
        FatherNode = Node()
        FatherNode.id = FreeID
        FreeID += 1

        # D = D \ {i*,j*}
        # V = V U {i*,j*} , E = E U {(f,i*),(f,j*)}

        for i in range(len(graph.leaves)): # for i*
            if (graph.leaves[i].id == Istar):
                graph.nodes.append(graph.leaves[i]) # V = V U {i*}
                graph.leaves[i].father = FatherNode  # E U {(f,i*)}
                FatherNode.children.append(graph.leaves[i]) # E U {(f,i*)}
                del graph.leaves[i] # D = D \ {i*}
                break

        for i in range(len(graph.leaves)): # for j*
            if (graph.leaves[i].id == Jstar):
                graph.nodes.append(graph.leaves[i]) # V = V U {j*}
                graph.leaves[i].father = FatherNode # E U {(f,j*)}
                FatherNode.children.append(graph.leaves[i]) # E U {(f,i*)}
                del graph.leaves[i] # D = D \ {j*}
                break

        # Step 2.2

        # d(s,f) = ρ(i*,j*)
        EstDistMatrix[0][FatherNode.id] = ScoreFunction[Istar][Jstar]
        EstDistMatrix[FatherNode.id][0] = EstDistMatrix[0][FatherNode.id] # SYMMETRY

        # d(f,i*) = d(s,i*) - ρ(i*,j*)
        EstDistMatrix[FatherNode.id][Istar] = EstDistMatrix[0][Istar] - ScoreFunction[Istar][Jstar]
        EstDistMatrix[Istar][FatherNode.id] = EstDistMatrix[FatherNode.id][Istar] # SYMMETRY

        # d(f,j*) = d(s,j*) - ρ(i*,j*)
        EstDistMatrix[FatherNode.id][Jstar] = EstDistMatrix[0][Jstar] - ScoreFunction[Istar][Jstar]
        EstDistMatrix[Jstar][FatherNode.id] = EstDistMatrix[FatherNode.id][Jstar] # SYMMETRY


        # Step 2.3
        # In this step we find if there are more than 2 siblings (if Istar,Jstar nodes have another sibling)
        #Calculate Δ based on the link parameter that is inferred
        delta = CalculateDelta()
        # For every k in D such that ρ(i*,j*) - ρ(i*,k) <= Δ/2:
        LeavesToDel = []
        for k in range (len(graph.leaves)):
            SiblID = graph.leaves[k].id # SiblID = node k ID
            if (ScoreFunction[Istar][Jstar] - ScoreFunction[Istar][SiblID] <= delta/2):
                # d(f,k) = d(s,k) - ρ(i*,j*)
                EstDistMatrix[FatherNode.id][SiblID] = EstDistMatrix[0][SiblID] - ScoreFunction[Istar][Jstar]
                EstDistMatrix[SiblID][FatherNode.id] = EstDistMatrix[FatherNode.id][SiblID] #SYMMETRY

                # D = D \ {k}
                # V = V U {k} , E = E U {(f,k)}
                graph.nodes.append(graph.leaves[k]) # V = V U {k}
                graph.leaves[k].father = FatherNode # E U {(f,k)}
                FatherNode.children.append(graph.leaves[k]) # E U {(f,k)}
                LeavesToDel.append(k) # D = D \ {k}, store the values to del together

        #create a temporary list that will have all the graph.leaves nodes besides those that we want to delete from step 2.3
        temp = []
        for i in range(len(graph.leaves)):
            if i not in LeavesToDel:
                temp.append(graph.leaves[i])
        graph.leaves = temp


        # Step 2.4

        for k in range(len(graph.leaves)):
            # d(k,f) = 1/2[d(k,i*)-d(f,i*)] + 1/2[d(k,j*)-d(f,j*)]
            EstDistMatrix[graph.leaves[k].id][FatherNode.id] = 0.5*(EstDistMatrix[graph.leaves[k].id][Istar]-EstDistMatrix[FatherNode.id][Istar]) + 0.5*(EstDistMatrix[graph.leaves[k].id][Jstar]-EstDistMatrix[FatherNode.id][Jstar])
            EstDistMatrix[FatherNode.id][graph.leaves[k].id] = EstDistMatrix[graph.leaves[k].id][FatherNode.id] #SYMMETRY

            # ρ(k,f) = 1/2[ρ(k,i*)+ρ(k,j*)]
            ScoreFunction[graph.leaves[k].id][FatherNode.id] = 0.5*(ScoreFunction[graph.leaves[k].id][Istar]+ScoreFunction[graph.leaves[k].id][Jstar])
            ScoreFunction[FatherNode.id][graph.leaves[k].id] = ScoreFunction[graph.leaves[k].id][FatherNode.id] # SYMMETRY


        # D = D U f
        graph.leaves.append(FatherNode)



    # If |D| = 1, for the i in D: V = V U {i} , E = E U (s,i)
    graph.nodes.append(graph.leaves[0])
    graph.leaves[0].father = graph.root
    graph.root.children = [graph.leaves[0]]

    # Draw the Topology produced by Tomography
    # variable "param" is used to draw the topology based on the specific inference parameter each time
    DrawTopology(param)

    # Write the results for each inference parameter performed in a csv file
    ExtractResults(param)
