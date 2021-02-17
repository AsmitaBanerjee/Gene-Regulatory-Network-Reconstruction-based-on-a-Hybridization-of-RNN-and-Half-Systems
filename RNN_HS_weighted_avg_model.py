"""
 Hybridized RNN-PSO Weighted Average Model

@author: asmita
"""

import numpy as np
from itertools import combinations
from itertools import permutations
from multiprocessing import Pool
import random
import sys
import time

if len(sys.argv)<4:
	print("Enter a txt input file name, a txt output file name and maxepoch after the program name")
	sys.exit(0)

#loading input data form txt file
def loadMatrix(file):
    inputData = np.genfromtxt(file,delimiter='\t')[1:,2:].transpose()
    return inputData

#initializing postition matrix
def initializePositionMatrix():
    position = []
    for item in combinations(np.arange(gene),connections):
        #list(item)
        position.append(list(item))
    #print(position)
    return position

#normalizing input data
def normalize(data1):
    maxi = np.amax(data1)
    mini = np.amin(data1)
    for item in data1:
        for i in range(len(item)):
            item[i] = (item[i] - mini)/(maxi-mini)
    return data1

#RnnHs #b@4 delta@5 t@6, mu@7
def RNNHSWeightedAverage(par,pos,inp,g):
    summ = par[connections]
    product = par[connections+1]
    s = 0.5
    for i in range(connections):
        par = np.insert(par,pos[i],0,axis=0)
    for i in range(gene):
        summ += par[i]*inp[i]
    if summ != 0:
        s = (1/(1+np.exp(-summ)))
    predV1 = ((deltaT/par[gene+2])*s)+((1-(deltaT/par[gene+2]))*inp[g])
    
    for i in range(gene):
        if(inp[i]!=0 and par[i]!= 0):
            product *= pow(inp[i],par[i]) 
    predV2 = (deltaT*product)+((1-par[gene+3])*inp[g])
    predV = 0.5*predV1 + 0.5*predV2

    return predV

#calculating MSE for RNNHSWeightedAverage
def MSEWeightedAverage(g,particleMatrix,positionMatrix):
    err = []
    for i in range(len(particleMatrix)):
        yTrue = np.copy(inputData[:,g])
        yTrue = yTrue[1:]
        yPred = []
        for j in range(len(inputData)-1):
            y = RNNHSWeightedAverage(particleMatrix[i],positionMatrix,inputData[j],g)
            yPred.append(y)
        err.append(np.square(np.subtract(yTrue,yPred)).mean())
    return err


#PSO inializaton
def PSO(g):
    
    positionMatrix = initializePositionMatrix()
    particleMatrix = np.random.uniform(-1,1,(population,connections+2))
    tau = np.random.uniform(low=1,high=tlim,size=(population,))
    mu = np.random.uniform(low=1/12,high=1,size=(population,))

    #inserting tau and mu
    particleMatrix = np.insert(particleMatrix, connections, mu, axis=1)
    particleMatrix = np.insert(particleMatrix, connections, tau, axis=1)  #b@4 delta@5 t@6, mu @7  
    
    
    velocityMatrix = np.zeros((population,connections+4))                          #v=(70x6)
    epoch = 0
    c = 0
    w = 0.95   #w varies from 0.95 to 0.45

    #initialization of pbest gbest
    error = MSEWeightedAverage(g, particleMatrix,positionMatrix)
    pBestError = np.copy(error)
    bestPosition = np.copy(particleMatrix)                       #bp=(70x6)
    gBestError = min(pBestError)
    minPos = list(pBestError).index(gBestError)
    gBest = np.copy(bestPosition[minPos])
    gBestPosition = np.copy(positionMatrix[minPos])

    while(epoch < maxEpoch):
        
        #velocity updation b@4 delta@5 t@6, mu@7
        for i in range(len(velocityMatrix)):
            
            for j in range(len(velocityMatrix[0])):
                r1 = random.uniform(0,1)
                r2 = random.uniform(0,1)
                velocityMatrix[i][j] = w*velocityMatrix[i][j] + c1*r1*(bestPosition[i][j]- particleMatrix[i][j]) + c2*r2*(gBest[j] - particleMatrix[i][j])
                if (j<connections+2):
                    if velocityMatrix[i][j]<-1:
                        velocityMatrix[i][j]=-1
                    elif(velocityMatrix[i][j]>1):
                        velocityMatrix[i][j]=1

                elif(j==connections+2):
                    if (velocityMatrix[i][j] <  1):
                        velocityMatrix[i][j] = 1
                    elif(velocityMatrix[i][j] > tlim):
                        velocityMatrix[i][j] = tlim
                    
                elif(j==connections+3):
                    if (velocityMatrix[i][j] <  1/12):
                        velocityMatrix[i][j] = 1 /12
                    elif(velocityMatrix[i][j] > 1):
                        velocityMatrix[i][j] = 1
                    

        #positionUpdation b@4 delta@5 t@6, mu@7
        for i in range(len(particleMatrix)):
            for j in range(particleMatrix[0]):
                particleMatrix[i][j] += velocityMatrix[i][j]
                if (j<connections+2):
                    if particleMatrix[i][j]<-1:
                        particleMatrix[i][j]=-1
                    elif(particleMatrix[i][j]>1):
                        particleMatrix[i][j]=1
                    
                elif(j==connections+2):
                    if (particleMatrix[i][j] <  1):
                        particleMatrix[i][j] = 1
                    elif(particleMatrix[i][j] > tlim):
                        particleMatrix[i][j] = tlim
                    
                elif(j==connections+3):
                    if (particleMatrix[i][j] <  1/12):
                        particleMatrix[i][j] = 1 /12
                    elif(particleMatrix[i][j] > 1):
                        particleMatrix[i][j] = 1          
            
        error = MSEWeightedAverage(g, particleMatrix,positionMatrix)
        
        #pBest updation
        for i in range(len(particleMatrix)):
            if (error[i] < pBestError[i]):
                pBestError[i] = error[i]
                bestPosition[i] = np.copy(particleMatrix[i])
        
        #gBest updation
        mini = min(pBestError)
        minPos = list(pBestError).index(mini)
        if (mini < gBestError or gBestError == -1):
            gBestError = mini
            gBest = np.copy(bestPosition[minPos])
            gBestPosition = np.copy(positionMatrix[minPos])
            c+=1
                
        epoch += 1
        w = w - (.5/maxEpoch)
    print("gbesterror[",g,"]: ",gBestError)
    print("counter[",g,"]:",c)
    return gBest, gBestPosition




#main
start = time.time()

gene = 8
connections = 4
population = 70
deltaT = 6
tlim = 2*deltaT
c1 = 2
c2 = 2
run = 10

inputData = loadMatrix(sys.argv[1])
actual = loadMatrix('Actual.txt')
inputData = normalize(inputData)
maxEpoch = int(sys.argv[3])
processor = int(input("Enter number of processors: "))

fp = tp = fn = tn = 0
fScore = 0
gBestAvg = np.empty((run,gene,gene+4))
avg = np.zeros((gene,gene))
add = np.zeros((gene,gene))

p = Pool(processor)


for r in range(run):
    gBest1 = p.map(PSO,range(gene))
    print("**********************************run: ",r+1,"************************************")
    gBest1 = np.asarray(gBest1)
    print("global best[",r+1,"]:\n",gBest1)
    for i in range(gene):
        for j in range(connections):
            gBest1[i][0] = np.insert(gBest1[i][0],gBest1[i][1][j],0,axis=0)
    for i in range(gene):
        for j in range(gene):
            if gBest1[i][0][j] != 0:
                add[i][j] += 1
    for i in range(len(gBest1)):
        gBestAvg[r][i] = np.copy(gBest1[i][0])

#number of connections
con = 0
#addition of all the connection matrices
print("Addition:\n", add)
threshold = int(input("Enter threshold: "))

for i in range(gene):
    for j in range(gene):
        if add[i][j] >= threshold:
            avg[i][j] = 1
            con += 1
for i in range(gene):
    for j in range(gene):
        if (avg[i][j] == 1 and actual[i][j] == 0):
            fp += 1
        elif (avg[i][j] == 1 and actual[i][j] == 1):
            tp += 1
        elif (avg[i][j] == 0 and actual[i][j] == 1):
            fn += 1
        else:
            tn += 1
if (tp+fp != 0):
    precision = tp/(tp+fp)
if (tp+fn != 0):
    recall = tp/(fn+tp)
if(precision != 0 or recall != 0):
    fScore = (2*((precision*recall)/(precision+recall)))
acc = (tp+tn)/(tp+tn+fp+fn)

print("Actual matrix:\n",actual,"\ngbestavg:\n",avg,"\nTotal connection:",con,"\nThreshold:",threshold)
print("\nTrue positive: ",tp,"\nFalse positive:",fp,"\nF score: ",fScore,"\nAccuracy: ",acc*100,"%")

stop = time.time()
print("Time taken: ",stop-start)

#writing the output to file
i=1
with open(sys.argv[2],'a') as f:
    f.write('*********************************RNN*************************************\nFile: rnn.v2.py\n')
    for item in gBestAvg:
        f.write('gbest[{0}]\n'.format(i))
        np.savetxt(f,item[:,:-4],fmt='%1.2f')
        i += 1
    f.write('Add\n')
    np.savetxt(f, add,fmt='%1.0f')
    f.write('Actual output\n')
    np.savetxt(f, actual,fmt='%1.0f')
    f.write('Final output\n')
    np.savetxt(f, avg,fmt='%1.0f')
    f.write('\nTotal number of connections: {0}\nThreshold: {1}'.format(con,threshold))
    f.write('\nTrue positive: {0}\nFalse positive: {1}\nTrue negative: {2}\nFalse negative: {3}'.format(tp,fp,tn,fn))
    f.write('\nRecall: {0}\nFallout: {1}\nSp: {2}\nPrecision: {3}\nF Score: {4}\nAccuracy: {5}%'.format((tp/(tp+fn)),(fp/(fp+tn)),(tn/(tp+tn)),(tp/(tp+fp)),fScore,acc*100))
    f.write('\ntime taken: {0}\nMax epoch: {1}'.format(stop-start,maxEpoch))