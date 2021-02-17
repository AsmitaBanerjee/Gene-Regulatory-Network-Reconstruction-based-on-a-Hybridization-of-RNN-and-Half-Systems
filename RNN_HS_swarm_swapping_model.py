"""
 Hybridized RNN-PSO Swarm Swapping model

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
    return np.asarray(position)


#normalizing input data
def normalize(data1):
    maxi = np.amax(data1)
    mini = np.amin(data1)
    for item in data1:
        for i in range(len(item)):
            item[i] = (item[i] - mini)/(maxi-mini)
    return data1

#RNN bias@4 tau@5
def RNN(par,pos,inp,g):
    summ = par[connections]
    s = 0.5
    for i in range(connections):
        par = np.insert(par,pos[i],0,axis=0)
    for i in range(gene):
        summ += par[i]*inp[i]
    if summ != 0:
        s = (1/(1+np.exp(-summ)))
    predV = ((deltaT/par[gene+1])*s)+((1-(deltaT/par[gene+1]))*inp[g])
    return predV

#calculating MSE
def MSErnn(g,particleMatrix,positionMatrix):
    err = []
    for i in range(len(particleMatrix)):
        yTrue = np.copy(inputData[:,g])
        yTrue = yTrue[1:]
        yPred = []
        for j in range(len(inputData)-1):
            y = RNN(particleMatrix[i],positionMatrix[i],inputData[j],g)
            yPred.append(y)
        err.append(np.square(np.subtract(yTrue,yPred)).mean())
    #print("err",err)
    return err

#HS delta@4, mu@5
def HS(par,pos,inp,g):
    #print(par[gene])
    product = par[connections]
    for i in range(connections):
        par = np.insert(par,pos[i],0,axis=0)
    #print ("***************",d)
    for i in range(gene):
        if(inp[i]!=0 and par[i]!= 0):
            product *= pow(inp[i],par[i]) 
    predV = (deltaT*product)+((1-par[gene+1])*inp[g])
    return predV

#calculating MSE for HS
def MSEhs(g,particleMatrix,positionMatrix):
    err = []
    for i in range(len(particleMatrix)):
        yTrue = np.copy(inputData[:,g])
        yTrue = yTrue[1:]
        yPred = []
        #d = 0
        for j in range(len(inputData)-1):
            y = HS(particleMatrix[i],positionMatrix, inputData[j],g)
            #d = yTrue[j]-y
            yPred.append(y)
        err.append(np.square(np.subtract(yTrue,yPred)).mean())
    return err


#PSO inializaton
def PSO(g):
    half = int(population/2)
    #initial population positions
    positionMatrix1 = initializePositionMatrix() #70*4
    #print(type(positionMatrix1), len(positionMatrix1))
    positionMatrix2 = np.copy(positionMatrix1[35:,:]) #35*4
    positionMatrix1 = np.copy(positionMatrix1[:35,:]) #35*4

    #initial populations
    particleMatrix1 = np.random.uniform(-1,1,(half,connections+1)) #35*5
    particleMatrix2 = np.random.uniform(-1,1,(half,connections+1)) #35*5
    
    #parameters
    tau = np.random.uniform(low=1,high=tlim,size=(half,1))
    mu = np.random.uniform(low=1/12,high=1,size=(half,1))
    
    
    
    #concatenating particle matrices with tau and mu
    particleMatrix1 = np.concatenate((particleMatrix1,tau),axis=1) #bias@4 tau@5 PM1 35*6
    particleMatrix2 = np.concatenate((particleMatrix2,mu),axis=1) #delta@4 mu@5 PM2 35*6
    
    
    velocityMatrix1 = np.zeros((half,connections+2))                          #v=(35x6)
    velocityMatrix2 = np.zeros((half,connections+2))                          #v=(35x6)
    
    swap = 0
       
    
    while(swap < 5):
        epoch = 0
        c1 = 0
        c2 = 0
        w = 0.95   #w varies from 0.95 to 0.45

        #initialization of pbest and gbest
        errorRNN = MSErnn(g,particleMatrix1,positionMatrix1)
        errorHS = MSEhs(g,particleMatrix2,positionMatrix2)
        pBestError1 = np.copy(errorRNN)
        pBestError2 = np.copy(errorHS)
        bestPosition1 = np.copy(particleMatrix1)                       #bp1=(70x6)
        bestPosition2 = np.copy(particleMatrix2)                       #bp2=(70x6)
        mini1 = min(pBestError1)
        minPos1 = list(pBestError1).index(mini1)
        gBestError1 = mini1
        gBest1 = np.copy(bestPosition1[minPos1])
        gBestPosition1 = np.copy(positionMatrix1[minPos1])
        mini2 = min(pBestError2)
        minPos2 = list(pBestError2).index(mini2)
        gBestError2 = mini2
        gBest2 = np.copy(bestPosition2[minPos2])
        gBestPosition2 = np.copy(positionMatrix2[minPos2])
        
        while (epoch < maxEpoch):
            #velocity updation b@4 t@5 / delta@4 mu@5
            for i in range(len(velocityMatrix1)):
                for j in range(len(velocityMatrix1[0])):
                    r1 = random.uniform(0,1)
                    r2 = random.uniform(0,1)
                    velocityMatrix1[i][j] = w*velocityMatrix1[i][j] + c1*r1*(bestPosition1[i][j]- particleMatrix1[i][j]) + c2*r2*(gBest1[j] - particleMatrix1[i][j])
                    if (j<connections+1):
                        if velocityMatrix1[i][j]<-1:
                            velocityMatrix1[i][j]=-1
                        elif(velocityMatrix1[i][j]>1):
                            velocityMatrix1[i][j]=1
                        
                    else:
                        if (velocityMatrix1[i][j] < 1):
                            velocityMatrix1[i][j] = 1
                        elif(velocityMatrix1[i][j] > tlim):
                            velocityMatrix1[i][j] = tlim
                        

                for j in range(len(velocityMatrix2[0])):
                    r1 = random.uniform(0,1)
                    r2 = random.uniform(0,1)
                    velocityMatrix2[i][j] = w*velocityMatrix2[i][j] + c1*r1*(bestPosition2[i][j]- particleMatrix2[i][j]) + c2*r2*(gBest2[j] - particleMatrix2[i][j])
                    if (j<connections+1):
                        if velocityMatrix2[i][j]<-1:
                            velocityMatrix2[i][j]=-1
                        elif(velocityMatrix2[i][j]>1):
                            velocityMatrix2[i][j]=1
                        
                    else:
                        if (velocityMatrix2[i][j] < 1/12):
                            velocityMatrix2[i][j] = 1/12
                        elif(velocityMatrix2[i][j] > 1):
                            velocityMatrix2[i][j] = 1
                        
                
            #positionUpdation b@4 tau@5/ delta@4, mu@5
            for i in range(len(particleMatrix1)):
                for j in range(len(particleMatrix1[0])):
                    particleMatrix1[i][j] += velocityMatrix1[i][j]
                    particleMatrix2[i][j] += velocityMatrix2[i][j]
                    if (j<connections+1):
                        if particleMatrix1[i][j]<-1:
                            particleMatrix1[i][j]=-1
                        elif(particleMatrix1[i][j]>1):
                            particleMatrix1[i][j]=1
                        if particleMatrix2[i][j]<-1:
                            particleMatrix2[i][j]=-1
                        elif(particleMatrix2[i][j]>1):
                            particleMatrix2[i][j]=1
                        
                    else:
                        if (particleMatrix1[i][j] < 1):
                            particleMatrix1[i][j] = 1
                        elif(particleMatrix1[i][j] > tlim):
                            particleMatrix1[i][j] = tlim
                        if (particleMatrix2[i][j] < 1/12):
                            particleMatrix2[i][j] = 1/12
                        elif(particleMatrix2[i][j] > 1):
                            particleMatrix2[i][j] = 1
                        
            errorRNN = MSErnn(g,particleMatrix1,positionMatrix1)
            errorHS = MSEhs(g,particleMatrix2,positionMatrix2)
            
            #pBest updation
            for i in range(len(particleMatrix1)):
                if (errorRNN[i] < pBestError1[i]):
                    pBestError1[i] = errorRNN[i]
                    bestPosition1[i] = np.copy(particleMatrix1[i])
                if (errorHS[i] < pBestError2[i]):
                    pBestError2[i] = errorHS[i]
                    bestPosition2[i] = np.copy(particleMatrix2[i])
            
            #gBest updation
            mini1 = min(pBestError1)
            minPos1 = list(pBestError1).index(mini1)
            mini2 = min(pBestError2)
            minPos2 = list(pBestError2).index(mini2)

            if (mini1 < gBestError1):
                gBestError1 = mini1
                gBest1 = np.copy(bestPosition1[minPos1])
                gBestPosition1 = np.copy(positionMatrix1[minPos1])
                c1+=1
            if (mini2 < gBestError2):
                gBestError2 = mini2
                gBest2 = np.copy(bestPosition2[minPos2])
                gBestPosition2 = np.copy(positionMatrix2[minPos2])
                c2+=1
                
            epoch += 1
            w = w - (.5/maxEpoch)
        
        temp = np.copy(particleMatrix2[:,:4])
        particleMatrix2[:,:4] = np.copy(particleMatrix1[:,:4])
        particleMatrix1[:,:4] = np.copy(temp)
        swap += 1
    
    print("gbesterror[",g,"]: ",gBestError1,gBestError2)
    print("counter[",g,"]:",c1,c2)
    return gBest1, gBestPosition1, gBest2, gBestPosition2




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

fp1 = tp1 = fn1 = tn1 = 0
fScore1 = 0
fp2 = tp2 = fn2 = tn2 = 0
fScore2 = 0
gBestAvg1 = np.empty((run,gene,gene+2))
gBestAvg2 = np.empty((run,gene,gene+2))
avg1 = np.zeros((gene,gene))
add1 = np.zeros((gene,gene))
avg2 = np.zeros((gene,gene))
add2 = np.zeros((gene,gene))

p = Pool(processor)


for r in range(run):
    #g = np.arange(gene)
    print("**********************************run: ",r+1,"************************************")
    gBest1 = p.map(PSO,range(gene))
    gBest1 = np.asarray(gBest1)
    print("global best[",r+1,"]:\n",gBest1)
    for i in range(gene):
        for j in range(connections):
            gBest1[i][0] = np.insert(gBest1[i][0],gBest1[i][1][j],0,axis=0)
            gBest1[i][2] = np.insert(gBest1[i][2],gBest1[i][3][j],0,axis=0)
    for i in range(gene):
        for j in range(gene):
            if gBest1[i][0][j] != 0:
                add1[i][j] += 1
            if gBest1[i][2][j] != 0:
                add2[i][j] += 1
    for i in range(len(gBest1)):
        gBestAvg1[r][i] = np.copy(gBest1[i][0])
        gBestAvg2[r][i] = np.copy(gBest1[i][2])

#number of connections
con1 = 0
con2 = 0
best = 2
#addition of all the connection matrices
print("Addition1:\n", add1)
print("Addition2:\n", add2)
stop = time.time()
threshold = int(input("Enter threshold: "))

for i in range(gene):
    for j in range(gene):
        if add1[i][j] >= threshold:
            avg1[i][j] = 1
            con1 += 1
        if add2[i][j] >= threshold:
            avg2[i][j] = 1
            con2 += 1
for i in range(gene):
    for j in range(gene):
        if (avg1[i][j] == 1 and actual[i][j] == 0):
            fp1 += 1
        elif (avg1[i][j] == 1 and actual[i][j] == 1):
            tp1 += 1
        elif (avg1[i][j] == 0 and actual[i][j] == 1):
            fn1 += 1
        else:
            tn1 += 1
        if (avg2[i][j] == 1 and actual[i][j] == 0):
            fp2 += 1
        elif (avg2[i][j] == 1 and actual[i][j] == 1):
            tp2 += 1
        elif (avg2[i][j] == 0 and actual[i][j] == 1):
            fn2 += 1
        else:
            tn2 += 1
if (tp1+fp1 != 0):
    precision = tp1/(tp1+fp1)
if (tp1+fn1 != 0):
    recall = tp1/(fn1+tp1)
if(precision != 0 or recall != 0):
    fScore1 = (2*((precision*recall)/(precision+recall)))
acc1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
if (tp2+fp2 != 0):
    precision = tp2/(tp2+fp2)
if (tp2+fn2 != 0):
    recall = tp2/(fn2+tp2)
if(precision != 0 or recall != 0):
    fScore = (2*((precision*recall)/(precision+recall)))
acc2 = (tp2+tn2)/(tp2+tn2+fp2+fn2)

if(fp1<fp2):
    best = 1


print("Actual matrix:\n",actual,"\ngbestavg1:\n",avg1,"\nTotal connection:",con1,"\nThreshold:",threshold)
print("\nTrue positive: ",tp1,"\nFalse positive:",fp1,"\nF score: ",fScore1,"\nAccuracy: ",acc1*100,"%")
print("Actual matrix:\n",actual,"\ngbestavg2:\n",avg2,"\nTotal connection:",con2,"\nThreshold:",threshold)
print("\nTrue positive: ",tp2,"\nFalse positive:",fp2,"\nF score: ",fScore2,"\nAccuracy: ",acc2*100,"%")
print("\nBest population: ",best)
finish = time.time()
print("\nTime taken before threshold calculation: ",stop-start)
print("\nTime taken for threshold calculation: ",finish-stop)


#writing the output to file
with open(sys.argv[2],'a') as f:
    f.write('File: rnn.v2.py\n*********************************RNN*************************************\n')
    i=1
    for item in gBestAvg1:
        f.write('gbest[{0}]\n'.format(i))
        np.savetxt(f,item[:,:-4],fmt='%1.2f')
        i += 1
    f.write('Actual output\n')
    np.savetxt(f, actual,fmt='%1.0f')
    f.write('Add\n')
    np.savetxt(f, add1,fmt='%1.0f')
    f.write('Final output\n')
    np.savetxt(f, avg1,fmt='%1.0f')
    f.write('\nTotal number of connections: {0}\nThreshold: {1}'.format(con1,threshold))
    f.write('\nTrue positive: {0}\nFalse positive: {1}\nTrue negative: {2}\nFalse negative: {3}'.format(tp1,fp1,tn1,fn1))
    f.write('\nRecall: {0}\nFallout: {1}\nSp: {2}\nPrecision: {3}\nF Score: {4}\nAccuracy: {5}%'.format((tp1/(tp1+fn1)),(fp1/(fp1+tn1)),(tn1/(tp1+tn1)),(tp1/(tp1+fp1)),fScore1,acc1*100))
    
    f.write('*********************************HS*************************************')
    i=1
    for item in gBestAvg2:
        f.write('gbest[{0}]\n'.format(i))
        np.savetxt(f,item[:,:-4],fmt='%1.2f')
        i += 1
    f.write('Add\n')
    np.savetxt(f, add2,fmt='%1.0f')
    f.write('Final output\n')
    np.savetxt(f, avg2,fmt='%1.0f')
    f.write('\nTotal number of connections: {0}\nThreshold: {1}'.format(con2,threshold))
    f.write('\nTrue positive: {0}\nFalse positive: {1}\nTrue negative: {2}\nFalse negative: {3}'.format(tp2,fp2,tn2,fn2))
    f.write('\nRecall: {0}\nFallout: {1}\nSp: {2}\nPrecision: {3}\nF Score: {4}\nAccuracy: {5}%'.format((tp2/(tp2+fn2)),(fp2/(fp2+tn2)),(tn2/(tp2+tn2)),(tp2/(tp2+fp2)),fScore2,acc2*100))
    f.write('\nBest population: {0}\n'.format(best))
    f.write('\n\Time taken for:\nBefore threshold calculation: {0}\nthreshold calculation: {1}'.format(stop-start,finish-stop))