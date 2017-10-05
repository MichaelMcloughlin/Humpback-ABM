####################################################

'''
Updates : 

Can specify which matrix goes to which breeding ground

Agents now don't discriminate. They move towards whatever agent that is singing nearby.

'distance'
'novelty'
'weightedEditsD'
'weightedEditsN'


'''

#Import libraries 
import numpy as np  
import difflib as dl 
import random 
import math 
import time
import pandas as pd 
from datetime import date
from collections import Counter 
import scipy.io as sio
import difflib as dl
import operator
import sys

#execfile('EncodeMultiFiles.py')

####################################################

class songModel(object):
   
    def __init__(self,maxSong = 100, #Maximum length of a song
                    r = 0.1, #Zone of repulsion
                    stopsong = 0.1, # zone of song interruption
                    width = 2, #width of our starting arena
                    height = 1000, #height of our starting arena
                    i = 12000, #number of iterations our model runs for
                    mRuns = 1, #The number of return migrations our model will carry out
                    iSave = 1, #how often to record data
                    MGS = 10, #Mating ground size 
                    FGS = 10,
                    FeedingGrounds = np.array([0,0]),
                    migrationTrigger = 1, #Migration trigger 
                    returnTrigger = 6000,
                    hearingThresh = 10, #CutoffDistance - How far our agents can hear a song before they can move towards it. 
                    probVec = np.array([ [0,1] , [2000,1], [6000,1],[8000,1] ]),
                    coinFlip = np.array([0.5,0.5]),
                    filename='runNumber_largeR',
                    inputFile = 'testbook.xlsx',
                    pL = np.array([0.8,0.1, 0.1]),
                    modelMode = 'distance'
                    ): # filename

        ####    These are global variable which should NOT be changed    ####
        #This is just used to measure how long it takes our model to run
        start_time = time.time()
        
        probVec = np.append(probVec,[[i+10,1]],axis = 0)
        #Get todays date (used for when we are saving data)
        today = date.today()
        
        #convert date to string
        today = str(today)
        
        set_sim_attributes(locals()) #set self attributes        
        
        params = locals()
        params.pop('self')
        #Dataframe for saving
        self.DF = {}
        self.DF['parameters'] = params 

        self.xcelInfo = pd.ExcelFile(self.inputFile)
        self.sheetNames = self.xcelInfo.sheet_names 
        self.AgentSetup = pd.read_excel(self.inputFile,'Sheet1', header = None)
        self.AgentSetup = np.array(self.AgentSetup)
        self.nw = len(self.AgentSetup[:,0])


        self.grammars = [0]*(len(self.sheetNames))
        self.noOfListenerAgents = self.nw  

        for n in xrange(1,len(self.sheetNames)):
            self.grammars[n] = pd.read_excel(self.inputFile,self.sheetNames[n],header = None)
            self.grammars[n] = np.array(self.grammars[n])

        self.grammars.pop(0)
        self.noOfUnits = len(self.grammars[0][0,:])

        
        #This keeps track of which agents changed their song
        self.listenerAgents = [] 
        
        self.Agents = []
        
        ####################################################


        #Set up agents 
        for x in xrange(0,self.nw):
            self.Agents.append(Agent(x,
                                     random.uniform(0,width), 
                                     random.uniform(0,height),
                                     self.grammars[self.AgentSetup[x][2]],
                                     i,
                                     self.nw,
                                     self.noOfUnits,
                                     r,
                                     stopsong,
                                     self.AgentSetup[x][0],
                                     self.AgentSetup[x][1],
                                     self.FeedingGrounds[0],
                                     self.FeedingGrounds[1],
                                     probVec,
                                     self.maxSong,
                                     self.pL,
                                     self.coinFlip ))
                    
            self.DF['whalenumber_' + str(x)] = {'name':x}


            
            
        #give them first songs and #print locations
        for a in self.Agents:
            a.singState = 1
            a.Sing(0)
            a.singState = 0

            print a.name
            print a.song
            #print a.location  
            
        self.run() #run the simulation
   
    def run(self): # MAIN MODEL LOOP           
        for numMig in xrange(0,self.mRuns):
            print 'this is migration number' + str(numMig)
            for n in xrange(0,self.i):
                print str(100*float(n)/self.i) + '%'
                
                #First calculate the distances of our agents
                for a in self.Agents:
                    a.distArray(self.Agents)
            
                #Movement Rules     
                for a in self.Agents:
                    a.seperate(self.Agents,self.r)
                    if n > self.migrationTrigger and n < self.returnTrigger:
                        a.seekOtherWhale(self.MGS,self.Agents,self.hearingThresh)

                    if n >= self.returnTrigger:
                        a.returnMigration(self.FGS)
                    
                    a.randDirection()
                    a.update()  
                                    
                    a.storeValue(n)
                    if a.stop_singing(self.Agents,self.stopsong) == 1:

                        a.Sing(n)
                        

                        if self.modelMode == ('weightedEditsD') or self.modelMode == ('weightedEditsN'):
                            a.editSong()

                        a.genPseudoGrammar()

                    #a.Sing(n)
        
            
                for m in xrange(0,self.nw):
                    #choose listener agent
                    listenerAgent = m
        
                    #calculate the intensity factors 
                    self.Agents[listenerAgent].calcIntFact()
        
                    #Calculate new grammar
                    self.Agents[listenerAgent].weightedGrammar(0,self.Agents,self.modelMode)
        
                    #This is used to keep track of our listener Agents
                    #listenerAgents.append(listenerAgent)           
             
                #Record data
                if np.remainder(n,self.iSave)==0:
                    for m in xrange(0,self.nw):
                        data = {'MigrationNumber' : numMig,
                                'X' : self.Agents[m].location[0],
                                'Y' : self.Agents[m].location[1], 
                                'distances' : self.Agents[m].distance, 
                                'IntFact' : self.Agents[m].IntFact, 
                                'song' : self.Agents[m].song,
                                'matrix' : self.Agents[m].mat,
                                'singStates': self.Agents[m].singState  } 

                        self.DF['whalenumber_' + str(m)]['iter_' + str(n)] = data
        
            #record endstate
            for m in xrange(0,self.nw):
                data = { 'MigrationNumber' : numMig,
                         'X' : self.Agents[m].location[0],
                         'Y' : self.Agents[m].location[1], 
                         'distances' : self.Agents[m].distance, 
                         'IntFact' : self.Agents[m].IntFact, 
                         'song' : self.Agents[m].song, 
                         'matrix' : self.Agents[m].mat, 
                         'singStates': self.Agents[m].singState }  

                self.DF['whalenumber_' + str(m)]['iter_' + str(n)] = data

            sio.savemat(self.filename+ 'migrationNumber' + str(numMig) + '.mat', self.DF)
        
        for a in self.Agents:
            print a.location

        print 'end'
        timeTaken = (time.time() - self.start_time)/60
        print 'the time taken to run the script was'
        print timeTaken
        for n in xrange(0,self.nw):
            print self.Agents[n].migX 
            print self.Agents[n].migY
            print self.Agents[n].mat
            print '---------'



####################################################

####                    Begin defining our class               #####  
####                    Change at your own peril               ##### 

class Agent(object):
    def __init__(self,name,X,Y,mat,i,nw,noOfUnits,r,stopsong,migX, migY,fgX,fgY,pVec,maxSong,pL,coinFlip):
        self.name = name
        set_sim_attributes(locals())
        #Variable associated with movement
        self.location = np.array([X,Y],dtype=np.float)
        self.velocity = np.array([0.1,0.1],dtype=np.float)
        self.acceleration = np.array([0.1,0.1],dtype=np.float)
        #THESE VALUES MUST ALWAYS BE FLOATS
        self.maxForce = 1.0
        self.maxspeed = 1.0
        self.trackX = [0] * i
        self.trackY = [0] * i

        self.migX = migX
        self.migY = migY

        self.fgX = fgX
        self.fgY = fgY 

        #Variables associated with song
        #How far our agent is from all the other agents 
        self.distance = [0] * nw 
        #Convert distance array to numpy array 
        self.distance = np.array(self.distance, dtype  = np.float)
        #The list to store the song
        self.song = []
        #Transition matrix. Will later become numpy matrix.
        self.mat = mat
        #Temporary storage for our pesudo gramar matrix
        self.pseudoMat = []
        #Our units dictionary 
        self.units = np.arange(0,noOfUnits)
        #Numpy array to store our intensity factors (we might need to set our own agents IntFact to zero...)
        self.IntFact = np.zeros(nw,dtype=np.float)
        #Boolean to test if whale is in mating ground or not. 
        self.IsMigrating = False
        #Song similarity index
        self.songindex = [0]*nw
        #self.songindex = np.array(self.songindex,dtype = np.float)
        self.singState = 0 #Sing state determines if an agent is singing or not. singState = 0 if not singing, 1 if singing
        self.probVec =  pVec
        self.noveltyValues = np.zeros(nw)
        self.maxSong = maxSong
        self.recursionPointer = 0 #This is a value to make sure we don't exceed our recursion depth
        self.pL = pL
        self.coinFlip = coinFlip
        self.pseudoOut = mat 
        

        

    #BEGIN MOVEMENT FUNCTIONS
    #Based on code from Daniel Shiffmans 'The Nature of Code'

    def applyForce(self,force):
        self.acceleration = self.acceleration + force

    def seperate(self,Agents,r):
        desiredSeperation = float(r) #float(r)*2
        tosum = np.array([0,0],dtype=np.float)
        count = 0
        
        for other in Agents:                    
            distance = dist(self.location[0],self.location[1],other.location[0],other.location[1])
            if distance > 0 and distance < desiredSeperation:
                difference = self.location - other.location
                #They mean make the distance of the vector one - Noramlize
                difference = difference/magnitude(difference[0],difference[1])
                difference = np.array(difference,dtype = np.float)
                difference = difference/distance
                tosum = tosum + difference
                count = count + 1

        if count > 0:
            tosum = tosum/float(count) 
            tosum = tosum/magnitude(tosum[0],tosum[1])
            tosum = tosum*self.maxspeed
            #'#print after we limit the maxspeed its'
            steering = tosum - self.velocity 
            steering = np.array(steering,dtype = np.float)
            ##print steering
            #Time to limit the value of steering
            steering = steering/magnitude(steering[0],steering[1])
            steering = steering*self.maxForce 
            self.applyForce(steering)


    def update(self):
        self.velocity = self.velocity + self.acceleration
        self.velocity = self.velocity/magnitude(self.velocity[0],self.velocity[1])
        self.velocity = self.velocity*self.maxspeed
        self.location = self.location + self.velocity
        self.acceleration = self.acceleration*0

    def seek(self, target):
        desired = target - self.location
        desired = desired/magnitude(desired[0],desired[1])
        desired = desired*self.maxspeed
        steer = desired - self.velocity
        steer = steer/magnitude(steer[0],steer[1])
        steer = steer*self.maxForce
        self.applyForce(steer)


    def storeValue(self,iter):
        self.trackX[iter] = self.location[0]
        self.trackY[iter] = self.location[1]

    def randWalk(self):
        angle = math.radians(random.randint(0,360))
        self.location[0] = self.location[0] + math.cos(angle)*3
        self.location[1] = self.location[1] + math.sin(angle)*3 

    def randDirection(self):
        diceRoll = random.randint(0,1)
        newDirection = np.array([0,0],dtype = np.float)
        if diceRoll == 1:
            angle = random.randint(0,360)
            newDirection[0] = math.cos(angle)
            newDirection[1] = math.sin(angle)

        self.applyForce(newDirection)

    def stop_singing(self, Agents, stopsong):
        stopsingSeperation = float(stopsong) 
        count = 0
        for other in Agents:                    
            distance = dist(self.location[0],self.location[1],other.location[0],other.location[1])
            if distance > 0 and distance < stopsingSeperation:
                count = count + 1
            if count > 0:
                self.singState = 0
                break 

        if count < 1:
            self.singState = 1
        #print self.singState
        #return self.singState
        #print self.singState
        #print '------------------------'
        return self.singState



    #END MOVEMENT FUCNTIONS
    def learnAgentGrammar(self,a,agentName,Agents,pseudoIn):
        
        
        if Agents[agentName].singState == 0:
            self.pseudoMat = np.zeros([self.noOfUnits,self.noOfUnits])
            #print 'I will not learn that grammar'
            return

        '''matSize = self.noOfUnits
        a = np.double(a)
        mat = hmmestimate(a,a)
        newMat = np.zeros([self.noOfUnits,self.noOfUnits])
        for n in xrange(0,self.noOfUnits):
            for m in xrange(0,self.noOfUnits):
                newMat[n,m] = mat[n,m]


        mat = newMat
        for n in xrange(0,self.noOfUnits):
            if sum(mat[n]) != 0:
                mat[n] = mat[n]/sum(mat[n])'''
        self.pseudoMat = pseudoIn
        return 

    def genPseudoGrammar(self):
        self.pseudoOut = hmmestimate(self.song,self.song)
        #if self.singState == 0:
        #    return

        




    #Sampling from our transition matrix. 
    def matrixSampling(self,choice):
        #print self.mat[choice]
        #If we get the final unit in our dictionary(the end unit), that signifies the end of the song. Break out of the function.
        if int(self.recursionPointer) == int(self.maxSong):
            self.recursionPointer = 0
            unitToAppend = self.noOfUnits
            self.song.append(unitToAppend-1)
            self.song = np.array(self.song) + 1
            self.song = list(self.song) 
            return 
        if choice == (self.noOfUnits-1):
            unitToAppend = self.noOfUnits  
            #np.random.choice(self.units,p = self.mat[choice-1])
            ##print unitToAppend
            self.song.append(unitToAppend-1)
            #self.song.append(unitToAppend)
            #Since the matrix sampling algorithm begins indexing at 0 (but the hmmestimate works from one...) 
            #We have to turn our sequence into a numpy array and add 1 so we don't have the unit 0 in our sequence
            self.song = np.array(self.song) + 1
            self.song = list(self.song)
            self.recursionPointer = 0
            return
            #self.noOfUnits = len(set(self.song))
        else: 
            #Here we pick a unit based on the transition vectors
            unitToAppend = np.random.choice(self.units,p = self.mat[choice])                        
            ##print unitToAppend
            self.song.append(unitToAppend)
            self.recursionPointer = self.recursionPointer + 1
            #This technique is called recursion. It's where you call a function inside itself. Its neat. 
            self.matrixSampling(unitToAppend)

    def Sing(self,iterNum):
        #Normalise Matrix
        #print self.mat

        if iterNum == 0:
            self.song = []
            unitToStart = 0
            self.song.append(unitToStart)
            self.matrixSampling(unitToStart)
            return



        p = 0
        for m in xrange(1,len(self.probVec)):
            if iterNum >= self.probVec[m-1,0] and iterNum <= self.probVec[m,0]:
                p = self.probVec[m-1,1]

        prob = np.array([abs(p-1),p])
        boolSing = np.random.choice([0,1], p = prob)
        #print boolSing
        if boolSing == 0:
            self.singState = 0
            #print 'I will not sing'
            return
        else:
            self.singState = 1
            #print 'I will sing'

        self.song = [] 
        unitToStart = 0
        self.song.append(unitToStart)
        self.matrixSampling(unitToStart)

    #This is used in a loop later to calculate the distance of each agent
    def distArray(self,Agents):
        for other in Agents:                    
            distance = dist(self.location[0],self.location[1],other.location[0],other.location[1])
            if 0<distance<1:
                distance = 1.0
            self.distance[other.name] = distance

    #Caclulate the intensity factors for all the other agents. 
    def calcIntFact(self):
        self.distance = np.array(self.distance, dtype = 'f')
        self.IntFact = 1.00/(self.distance**2) #spherical spreading**2
        #We make our own agents intensity factor 0 so we don't end up adding its grammar on top of itself
        self.IntFact[int(self.name)] = 0
        self.IntFact = np.array(self.IntFact,dtype = 'f')
        ##print self.IntFact


    def weightedGrammar(self,agentNumber,Agents,modelMode):
        #Calculate the novelty of all agents songs
        if (modelMode == 'novelty') or modelMode == ('weightedEditsN'):
            self.allNovelty(Agents)

        #Learn the grammar of all the agents. We start with agent 0 
        for a in Agents:
            if a.name != self.name:

                #self.learnAgentGrammar(a.song,a.name,Agents,a.pseudoOut)
                G2 = a.pseudoOut

                if Agents[a.name].singState == 0:
                    #print 'I will not learn the song'
                    continue

                #print 'I will learn the song'
                
                if (modelMode == 'distance') or (modelMode == 'weightedEditsD') :
                    scalar = 1 - self.IntFact[a.name]
                    scalar2 = (self.IntFact[a.name])
                    self.mat = self.mat*(scalar) + G2*scalar2

                if (modelMode == 'novelty') or modelMode == ('weightedEditsN'):
                    scalar = 1 - (self.IntFact[a.name]*self.noveltyValues[a.name])
                    scalar2 = (self.IntFact[a.name]*self.noveltyValues[a.name])
                    self.mat = (self.mat*(scalar) +  G2*scalar2)
              

            
        for n in xrange(0,len(self.mat)):
            if sum(self.mat[n]) != 0:
                self.mat[n] = self.mat[n]/sum(self.mat[n])




        return

    def allNovelty(self,Agents):
        for a in Agents:
            if a.name != self.name:
                self.noveltyValues[a.name] = novelty(a.song,self.mat)
            else:
                self.noveltyValues[a.name] = 0
        
        #self.noveltyValues = np.array(self.noveltyValues, dtype = 'float')

        #if sum(self.noveltyValues) != 0:
            #self.noveltyValues = self.noveltyValues/sum(self.noveltyValues)

    
       
        #self.noveltyValues = a + (b-a)*self.noveltyValues


    def seekOtherWhale(self,MGS,Agents,hearingThresh):


        distScores = {}

        #Check how far we are from the migration zone. If we are outside it move towards it. Otherwise, seekwhales
        MGdistance = dist(self.location[0],self.location[1],self.migX,self.migY)
        if MGdistance < MGS:
            for other in Agents:
            	#print 'state for the agent is'
            	#print other.singState
            	if int(other.singState) == 0:
            		#print 'I will not follow that whale'
            		##print 'its oay mike'
            		continue
            	else:
                     #print 'I will follow that whale'
                     distance = dist(self.location[0],self.location[1],other.location[0],other.location[1])
                     if self.name == other.name:
                        continue
                     ##print distance
                     if distance < hearingThresh:
                         distScores[other.name] = distance 

            if bool(distScores) == False:
                return 
           

          
            agentWithHighestScore = keywithminval(distScores)
            

            if agentWithHighestScore == self.name:
                'I am the agent who is closest. Impossible. Just go centre of mating ground.'
                self.seek([self.migX,self.migY])
            else:
                self.seek(Agents[agentWithHighestScore].location)
        else:
            self.seek([self.migX,self.migY])

    def returnMigration(self,FGS):
        FGdistance = dist(self.location[0],self.location[1],self.fgX,self.fgY)
        if FGdistance > FGS:
            self.seek([self.fgX,self.fgY])

    def editSong(self):
        #print 'time to edit the song'
        coinFlipV = np.random.choice([0,1], p = self.coinFlip)
        editChoice = np.random.choice([1,2,3], p = self.pL)
        if len(self.song) == 0 or len(self.song) == 1:
            editPoint = 0
        else:
            editPoint = np.random.randint(1,len(self.song))
        chosenUnit = np.random.choice(self.units,p = np.repeat((1.0/self.noOfUnits),self.noOfUnits))
        if coinFlipV == 1:
            #print 'yas queen'

            if editChoice == 1:
                #Carry out insertion of a random unit
                #print 'I have carried out insertion'
                self.song.insert(editPoint,chosenUnit)

            if editChoice == 2:
                #print 'I have done a deletion'
                if len(self.song) < 2:
                    return
                del(self.song[editPoint])

            if editChoice == 3:
                #print 'I have done a substiution'
                self.song[editPoint] = chosenUnit






####################################################

####    These are functions that need to be declared    ####
####                DO NOT CHANGE THESE                 ####

#helper function to avoid self.* = * boilerplate
def set_sim_attributes(d): 
    self=d.pop('self')
    for n,v in d.iteritems():
        setattr(self,n,v) 

#function to measure distance between two caretesian points
def dist(x1,y1,x2,y2):
    return math.sqrt( ( float(x1)-float(x2) )**2 + ( float(y1)-float(y2) )**2 )

#Function to measure the magnitude of a vector 
def magnitude(X,Y):
    return math.sqrt((float(X)**2 + float(Y)**2))

#This funtion is used to calculate a transition matrix. 
def tmatrix(a):
    matSize = noOfUnits
    b = np.zeros([matSize,matSize])
    for (x,y), c in Counter(zip(a, a[1:])).iteritems():
        b[x-1,y-1] = c
    b = b/b.sum(axis=1, keepdims=True)
    np.nan_to_num(b)
    return b


def keywithmaxval(d):
    """ a) create a list of the dict's keys and values; 
        b) return the key with the max value"""  
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]

def keywithminval(d):
    """ a) create a list of the dict's keys and values; 
        b) return the key with the min value"""  
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(min(v))]


def keywithsecondminval(d):
    """ a) create a list of the dict's keys and values; 
        b) return the key with the second lowest value""" 
    if len(d) == 1:
        d = (d.keys())
        return d[0]

    v=list(d.values())
    k=list(d.keys())

    a = list(v)

    b =  list(k)

    k.pop(v.index(min(v)))

    v.pop(v.index(min(v)))

    test = k[v.index(min(v))]

    return test

def hmmestimate(seq,states):
    """Python implementation of Matlab hmmestimate function"""
    numStates = max(states) 
    seqLen = len(seq) 
    tr = np.zeros((numStates,numStates))

    # count up the transitions from the state path
    for count in range(seqLen-1):
        tr[states[count]-1,states[count+1]-1] += 1

    trRowSum = np.array(np.sum(tr,1))

    #% if we don't have any values then report zeros instead of NaNs.
    trRowSum[trRowSum == 0] = -np.inf

    #% normalize to give frequency estimate.
    tr = tr/np.tile(trRowSum[:,None],numStates)

    return tr

def novelty(seq,matrix):
    '''Novelty distance measurement'''
    N = len(seq)
    scoreVals = np.zeros(len(seq))
    seq = np.array(seq)
    seq = seq - 1
    for n in xrange(0,N-1):
        heardUnitScore = matrix[ seq[n], seq[n+1] ]
        expectedUnitScore = max(matrix[seq[n],:])
        scoreVals[n] = expectedUnitScore - heardUnitScore
    output = np.sum(scoreVals)
    output = output/(N)
    return output

def secondLowestIndex(input):
    a = list(input)
    b = list(input)

    b.pop(b.index(min(b)))

    valueToGet = min(b)

    output = a.index(valueToGet)

    return output












        




