####################################################

'''
This model was developed for the Music and Science Paper titled 

"Using agent-based models to understand the role of individuals in the song evolution of humpback whales (Megaptera novaeangliae)."

Corresponding author : michaelmcl1991@gmail.com 

Model designed by : Michael Mcloughlin and Luca Lamoni with input from Luke Rendell, Ellen Garland, Alexis Kirke, Simon Ingram, Mike Noad, and Eduardo Miranda

This code is provided for educational purposes. The authors accept no responsibility for any damages caused by this model. 
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

####################################################

#Our model class
class songModel(object):
   
    def __init__(self,maxSong = 100, #Maximum length of a song
                    r = 0.1, #Zone of repulsion
                    stopsong = 0.1, # zone of song interruption
                    width = 2, #width of our starting arena
                    height = 1000, #height of our starting arena
                    i = 12001, #number of iterations our model runs for
                    mRuns = 2, #The number of return migrations our model will carry out
                    iSave = 100, #how often to record data
                    MGS = 100, #Mating ground size 
                    FGS = 50, #Feeding ground size 
                    FeedingGrounds = np.array([0,0]), #The coordinates of the feeding grounds 
                    migrationTrigger = 1, #Migration trigger 
                    returnTrigger = 6000, #Controls what iteration the agents return to the feeding grounds 
                    hearingThresh = 10, #CutoffDistance - How far our agents can hear a song before they can move towards it. 
                    probVec = np.array([ [0,0.5] , [2000,0.8], [6000,0.5],[8000,0.08] ]),
                    coinFlip = np.array([0.99,0.01]), #Controls the liklihood of a levenshtein distance edit being carried out
                    filename='runNumber_largeR', #Name of the matlab file to be outputby the model 
                    inputFile = 'testbook.xlsx', #Name of the input Excel file. 
                    pL = np.array([0.8,0.1, 0.1]), #The probability of carrying out either an insertion (first element), deletion (second element), or substiution(third element)
                    modelMode = 'distance' #The type of model we are running. 'distance' = model 1 from paper, 'novelty' = model 2 from paper, 
                                           #'weightedEditsD'= model 3 from paper, 'weightedEditsN' = model 4 from paper
                    ): # filename

        ####    These are global variable which should NOT be changed    ####
        #This is just used to measure how long it takes our model to run
        start_time = time.time()
        
        #Empty array that will later be used to control the likelihood of an agent singing
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
        
        #######   Excel stuff  #########

        #Grab the excel input file
        self.xcelInfo = pd.ExcelFile(self.inputFile)

        #Get the sheet names
        self.sheetNames = self.xcelInfo.sheet_names 

        #This is used to store the sheets that determine an agents input SR
        self.AgentSetup = pd.read_excel(self.inputFile,'Sheet1', header = None)

        #Transform sheets into numpy arrays
        self.AgentSetup = np.array(self.AgentSetup)

        #Get the number of whales from the excel sheets
        self.nw = len(self.AgentSetup[:,0])

        #emptry array to store the SR's in a seperate list 
        self.grammars = [0]*(len(self.sheetNames))

        #Set the number of agents that can listen per turn (redundant variable that will be removed in later versions)
        self.noOfListenerAgents = self.nw  
 

        #store SR's from excel into seperate list
        for n in xrange(1,len(self.sheetNames)):
            self.grammars[n] = pd.read_excel(self.inputFile,self.sheetNames[n],header = None)
            self.grammars[n] = np.array(self.grammars[n])

        
        #Remove the first excel sheet from the grammars variable (this is the data used to assign breeding ground co-ordinates and SR's)
        self.grammars.pop(0)

        #Get the number of units for the song from the SR's. 
        self.noOfUnits = len(self.grammars[0][0,:])

        
        #This keeps track of which agents changed their song
        self.listenerAgents = [] 
        
        #Empty list that stores our agents
        self.Agents = []
        
        ####################################################


        #Set up agents 
        for x in xrange(0,self.nw):
            #Create an agent and store it in the agents list
            self.Agents.append(Agent(x,
                                     random.uniform(self.AgentSetup[x][3],self.AgentSetup[x][4]), 
                                     random.uniform(self.AgentSetup[x][3],self.AgentSetup[x][4]),
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
            
            #Store the data in a dataframe to save later        
            self.DF['whalenumber_' + str(x)] = {'name':x}


            
            
        #give them first songs and #print locations
        for a in self.Agents:
            #Agents need to sing initially so that their initial song variables are not empty when the model starts
            #This can cause problems if they are. We then turn the singstate back off and allow it to be controleld
            #by the other variables that control singing
            a.singState = 1
            a.Sing(0)
            a.singState = 0

            print a.name
            print a.song
            #print a.location  
            
        self.run() #run the simulation
   
    def run(self): # MAIN MODEL LOOP  
        #This loop keeps track of what migration we are currently going through          
        for numMig in xrange(0,self.mRuns):
            print 'this is migration number' + str(numMig)

            #This loop is the actual migration that the agents undergo 
            for n in xrange(0,self.i):

                #Calculates and prints what percentage of the model is complete
                print str(100*float(n)/self.i) + '%'
                
                #First calculate the distances of our agents
                for a in self.Agents:
                    a.distArray(self.Agents)
            
                #Movement Rules   

                #loop through every agent  
                for a in self.Agents:
                    #current agent carried out seperation function (zor function)
                    a.seperate(self.Agents,self.r)
                    #Check to see what stage of the migration we are at. If we are currently in the middle of migration, seek a whale
                    #Please note that the seekOtherWhale funtion has a conditional that prevents this from being carried out if the 
                    #agents are not on the breeding grounds
                    if n > self.migrationTrigger and n < self.returnTrigger:
                        #Seek other whale
                        a.seekOtherWhale(self.MGS,self.Agents,self.hearingThresh)
                    #If the return migration trigger is active, we return to the feeding grounds
                    if n >= self.returnTrigger:
                        a.returnMigration(self.FGS)
                    
                    #Add noise to the agents trajectors
                    a.randDirection()
                    #Update the agents trajectors
                    a.update()  
                    
                    #Store its caretesian co-ordinates for saving (you can use these to plot their co-ordinates later if you want)                
                    a.storeValue(n)

                    #Check if agent is supposed to be singing during this iteration. if it is then proceed with conditional
                    if a.stop_singing(self.Agents,self.stopsong) == 1:
                        
                        #Agent produces a song
                        a.Sing(n)
                        
                        #If we are using weighted edits (either distance (model 3) or novelty (model 4) then edit the song
                        if self.modelMode == ('weightedEditsD') or self.modelMode == ('weightedEditsN'):
                            a.editSong()
                        
                        #Generate the pseudo grammar. This is calculated now so that each agent does not need to calculate, thus significantly reducing run time. 
                        a.genPseudoGrammar()

        
                
                #The agents learn their songs here 
                for m in xrange(0,self.nw):

                    #choose listener agent
                    listenerAgent = m
        
                    #calculate the intensity factors 
                    self.Agents[listenerAgent].calcIntFact()
        
                    #Calculate new grammar
                    self.Agents[listenerAgent].weightedGrammar(0,self.Agents,self.modelMode)
        
                    #This is used to keep track of our listener Agents (obsolete)
                    #listenerAgents.append(listenerAgent)           
             
                #Record  and save data
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
            #Obsolete, originally used to track an agents location at the end of arun.
            print a.location

        print 'end'

        #Print how long the model took to run 
        timeTaken = (time.time() - self.start_time)/60
        print 'the time taken to run the script was'
        print timeTaken
        #for n in xrange(0,self.nw):
            #print self.Agents[n].migX 
            #print self.Agents[n].migY
            #print self.Agents[n].mat
            #print '---------'



####################################################

####                    Begin defining our agent class               #####  
####                    Change at your own peril!!!!!!               ##### 

class Agent(object):
    def __init__(self,name,X,Y,mat,i,nw,noOfUnits,r,stopsong,migX, migY,fgX,fgY,pVec,maxSong,pL,coinFlip):
        #Agents name (helps keep track of them inside the agent list variable)
        self.name = name
        #Assign universal variables (the ones that are not input to the model via the excel input file)
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
        
        #Migration co-ordinates for the agents
        self.migX = migX
        self.migY = migY

        #Feeding ground co-ordinates associated with the agents. 
        self.fgX = fgX
        self.fgY = fgY 

        #Variables associated with song
        #How far our agent is from all the other agents 
        self.distance = [0] * nw 
        #Convert distance array to numpy array 
        self.distance = np.array(self.distance, dtype  = np.float)
        #The list to store the song an agent produces
        self.song = []
        #Transition matrix. Will later become numpy matrix.
        self.mat = mat
        #Temporary storage for our pesudo gramar matrix
        self.pseudoMat = []
        #Our units dictionary (all units available to an agent. Important for editing the song)
        self.units = np.arange(0,noOfUnits)
        #Numpy array to store our intensity factors (we might need to set our own agents IntFact to zero...)
        self.IntFact = np.zeros(nw,dtype=np.float)
        #Boolean to test if whale is in mating ground or not. 
        self.IsMigrating = False
        #Song similarity index
        self.songindex = [0]*nw
        #self.songindex = np.array(self.songindex,dtype = np.float)
        self.singState = 0 #Sing state determines if an agent is singing or not. singState = 0 if not singing, 1 if singing
        self.probVec =  pVec #Probability vector controls whether or not an agent sings at a given iterations
        self.noveltyValues = np.zeros(nw) #stores the novelty values calculated by an agent. Each index corresponds to agents name (so index 0 corresponds to agent 0)
        self.maxSong = maxSong #The maximum length a song can be
        self.recursionPointer = 0 #This is a value to make sure we don't exceed our recursion depth
        self.pL = pL #variable used to contro Probability of carrying out a specific levenshtein distance edit
        self.coinFlip = coinFlip #The probabiliity of an edit being carried out.
        self.pseudoOut = mat #The estimated output of an agents song (see line 198)
        

        

    #BEGIN MOVEMENT FUNCTIONS
    #This is basically a python port of code from Daniel Shiffmans 'The Nature of Code'. 

    def applyForce(self,force):
        self.acceleration = self.acceleration + force

    def seperate(self,Agents,r):
        #The desired separation is our zor
        desiredSeperation = float(r) #float(r)*2
        #This is used to calculate the new trajectory the agent will take later
        tosum = np.array([0,0],dtype=np.float)
        #The number of agents in the zor
        count = 0
        
        #Loop through every agent
        for other in Agents:   
            #Calculate distance                 
            distance = dist(self.location[0],self.location[1],other.location[0],other.location[1])
            #If distance between agents is within the ZOR, account for this agent in calculating a new trajectory
            if distance > 0 and distance < desiredSeperation:
                #Calculate the difference between the two agents location
                difference = self.location - other.location
                #This mean make the distance of the vector one - IE Noramlize
                difference = difference/magnitude(difference[0],difference[1])
                #Make sure it's a float before carrying out division
                difference = np.array(difference,dtype = np.float)
                #Divide the difference by the distance between the agents
                difference = difference/distance
                #Sum the values of all agents within the zor up to now
                tosum = tosum + difference
                #move on to the next agent and keep track of how many are within the ZOR
                count = count + 1

        #If are  agents in the ZOR
        if count > 0:
            #Divide the summed values by number of agents in the zor
            tosum = tosum/float(count) 
            #Divide by the total magnitide
            tosum = tosum/magnitude(tosum[0],tosum[1])
            #This means our agents cannot excess the maxspeed variable 
            tosum = tosum*self.maxspeed
            #'#print after we limit the maxspeed its'
            #Calculate the new trajector by subtracting the tosum value from the agents current velocity
            steering = tosum - self.velocity 
            steering = np.array(steering,dtype = np.float)
            ##print steering
            #Tlimit the value of steering
            steering = steering/magnitude(steering[0],steering[1])
            steering = steering*self.maxForce 
            #Push the agent in the direction of the steering value 
            self.applyForce(steering)

    #update the cartesian coordinates based on the velocity 
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

    #Obsolete function. Will be deleted in future versions
    def randWalk(self):
        angle = math.radians(random.randint(0,360))
        self.location[0] = self.location[0] + math.cos(angle)*3
        self.location[1] = self.location[1] + math.sin(angle)*3 

    #Add noise to the agents direction
    def randDirection(self):
        diceRoll = random.randint(0,1)
        newDirection = np.array([0,0],dtype = np.float)
        if diceRoll == 1:
            angle = random.randint(0,360)
            newDirection[0] = math.cos(angle)
            newDirection[1] = math.sin(angle)

        self.applyForce(newDirection)

    #If an agent has another agent within a certain radius, stop singing. In our experiments this is just set to the ZOR value. 
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

    #This function is where an agent "learns" a song (see Figure 1 of Music and Science paper)
    def learnAgentGrammar(self,a,agentName,Agents,pseudoIn):
        
        #Check to see if an agent is singing. If they are not singing, do not learn their song
        if Agents[agentName].singState == 0:
            #Assign a value of zero that pseudomatrix vale (pseudomat variables cannot be empty)
            self.pseudoMat = np.zeros([self.noOfUnits,self.noOfUnits])
            #print 'I will not learn that grammar'
            return

        #Assign agents pseudomatrix to variabel
        self.pseudoMat = pseudoIn
        return 

    #Generate output SR (estimate transition matrix for agents own song)
    def genPseudoGrammar(self):
        self.pseudoOut = hmmestimate(self.song,self.song)
        #if self.singState == 0:
        #    return



    #Sampling from our transition matrix. (basically where our agents are singing)
    def matrixSampling(self,choice):

        #If we go past our recursion depth, break out of the function.
        if int(self.recursionPointer) == int(self.maxSong):
            #Reset recursion Pointer
            self.recursionPointer = 0
            #Append final unit
            unitToAppend = self.noOfUnits
            #Append again so that transition from final unit always has a 100% probability 
            self.song.append(unitToAppend-1)
            #This is done because of how the hmmestimate function works. 
            self.song = np.array(self.song) + 1
            #Transform frrom array to list 
            self.song = list(self.song) 
            return 
    

        #If we get the final unit in our dictionary(the end unit), that signifies the end of the song. Break out of the function.
        if choice == (self.noOfUnits-1):
            #Append whatev
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
            #Here we pick a unit based on the transition matrix (SR)
            unitToAppend = np.random.choice(self.units,p = self.mat[choice])                        
            ##print unitToAppend
            self.song.append(unitToAppend)
            self.recursionPointer = self.recursionPointer + 1
            #This technique is called recursion. It's where you call a function inside itself. Its neat. 
            self.matrixSampling(unitToAppend)

    def Sing(self,iterNum):

        #If this is the first iteration, we need to initialise the song as follows 
        if iterNum == 0:
            self.song = []
            unitToStart = 0
            self.song.append(unitToStart)
            self.matrixSampling(unitToStart)
            return


        #This p variable is used in a coinflip to determine whether or not an agent sings based on the agents probVec variable
        p = 0
        for m in xrange(1,len(self.probVec)):
            #Check where we are in the mode and if we need to update the singing probability based on the provVec variable
            if iterNum >= self.probVec[m-1,0] and iterNum <= self.probVec[m,0]:
                p = self.probVec[m-1,1]

        #Caclulate probability 
        prob = np.array([abs(p-1),p])

        #Coinflip for whether they will sing or not
        boolSing = np.random.choice([0,1], p = prob)
        #If coin is tails don't sing (singState variable = 0)
        if boolSing == 0:
            self.singState = 0
            #print 'I will not sing'
            return
        else:
            #If coin is heads then sing (singState variable is 1)
            self.singState = 1
            #print 'I will sing'

        #Delete song from previous iteration
        self.song = []
        #Start on unit 0 
        unitToStart = 0
        #Attch the unit to the song array
        self.song.append(unitToStart)
        #Sample from the transition matrix (SR)
        self.matrixSampling(unitToStart)

    #This is used in a loop later to calculate the distance of each agent (basically just pythragoras theorem)
    def distArray(self,Agents):
        for other in Agents:                    
            distance = dist(self.location[0],self.location[1],other.location[0],other.location[1])
            if 0<distance<1:
                distance = 1.0
            self.distance[other.name] = distance

    #Caclulate the intensity factors for all the other agents. 
    def calcIntFact(self):
        self.distance = np.array(self.distance, dtype = 'f') #Making sure the array we store our values in are floats
        self.IntFact = 1.00/(self.distance**2) #spherical spreading**2
        #We make the agent carrying out this calculations intensity factor 0 so we don't end up adding its grammar on top of itself.
        self.IntFact[int(self.name)] = 0
        self.IntFact = np.array(self.IntFact,dtype = 'f')
        ##print self.IntFact


    def weightedGrammar(self,agentNumber,Agents,modelMode):
        #Calculate the novelty of all agents songsn if we are in a novelty mode
        if (modelMode == 'novelty') or modelMode == ('weightedEditsN'):
            self.allNovelty(Agents)

        #Learn the grammar of all the agents. We start with agent 0 
        for a in Agents:
            if a.name != self.name:

                #self.learnAgentGrammar(a.song,a.name,Agents,a.pseudoOut)
                G2 = a.pseudoOut

                #Check if that agent is singing. If they're not, skip that agent. 
                if Agents[a.name].singState == 0:
                    #print 'I will not learn the song'
                    continue

                
                #Check which model we are running. If it's distance or distance and weighted edits, don't carry out novelty calculation.
                if (modelMode == 'distance') or (modelMode == 'weightedEditsD') :
                    #See Figure 1 of the paper. 
                    scalar = 1 - self.IntFact[a.name]
                    scalar2 = (self.IntFact[a.name])
                    self.mat = self.mat*(scalar) + G2*scalar2

                #if we are running one of the novelty models, carry out a novelty weighting calculation
                if (modelMode == 'novelty') or modelMode == ('weightedEditsN'):
                    scalar = 1 - (self.IntFact[a.name]*self.noveltyValues[a.name])
                    scalar2 = (self.IntFact[a.name]*self.noveltyValues[a.name])
                    self.mat = (self.mat*(scalar) +  G2*scalar2)
              

        #Normalise matrices            
        for n in xrange(0,len(self.mat)):
            if sum(self.mat[n]) != 0:
                self.mat[n] = self.mat[n]/sum(self.mat[n])




        return

    #Calculate the novelty for every agent
    def allNovelty(self,Agents):
        #loop through every agent
        for a in Agents:
            #conditional to make sure you don't calculate your own novelty
            if a.name != self.name:
                #Calculate novelty and store in array 
                self.noveltyValues[a.name] = novelty(a.song,self.mat)
            else:
                #Make sure your own novelty value is equal to zero 
                self.noveltyValues[a.name] = 0
        

    #This is used to cause a whale to seek the nearest whale to them on the breeding ground. 
    def seekOtherWhale(self,MGS,Agents,hearingThresh):

        #Dictionary to store the distance of every agent from the current agent carrying out this function. 
        distScores = {}

        #Check how far we are from the migration zone. If we are outside it move towards it. Otherwise, seekwhales
        MGdistance = dist(self.location[0],self.location[1],self.migX,self.migY)
        
        #If we are inside the mating grounds
        if MGdistance < MGS:
            #Loop through every agent
            for other in Agents:
            	#Check to see if current agent is singing. If it is not singing follow it. 
            	if int(other.singState) == 0:
            		#print 'I will not follow that whale'
            		##print 'its oay mike'
            		continue
            	else:
                     #Calculate the distance
                     distance = dist(self.location[0],self.location[1],other.location[0],other.location[1])

                     #To make sure an agent does not calculate distance between themself
                     if self.name == other.name:
                        continue
                     #Check to see if the distance is within the zone of attraction, if it is, store it in the distscores variable
                     if distance < hearingThresh:
                         distScores[other.name] = distance 

            if bool(distScores) == False:
                #Safety check. If something goes wrong with our distScores variable we break out of the function 
                return 
           

            #Find the closest agent to the agent currently carrying out this calculation 
            agentWithHighestScore = keywithminval(distScores)
            
            #Extra safety check: Make sure you're not calculating your own distance. 
            if agentWithHighestScore == self.name:
                'I am the agent who is closest. Impossible. Just go centre of mating ground.'
                self.seek([self.migX,self.migY])
            else:
                #Move towards the closest agent
                self.seek(Agents[agentWithHighestScore].location)
        else:
            #If outside the mating ground move towards it
            self.seek([self.migX,self.migY])

    #This function forces the agents to return to the feeding ground. 
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


#Find the key value in a dictionary
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


#Calculate our transition matrix. Based on the matlab hmmestimate function. 
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


#Used to calculate novelty
def novelty(seq,matrix):
    '''Novelty distance measurement'''
    #Sequence length
    N = len(seq)
    #Array to store novelty scores
    scoreVals = np.zeros(len(seq))
    #Input sequence
    seq = np.array(seq)
    #Adjust for indexing
    seq = seq - 1
    #Loop through sequence
    for n in xrange(0,N-1):
        #Get the transition value from inputSequence[n,n+1]
        heardUnitScore = matrix[ seq[n], seq[n+1] ]
        #Get the most highest value associate with the transition from inputSequence[n] 
        expectedUnitScore = max(matrix[seq[n],:])
        #Calcualte difference between expected transition and the heard transition
        scoreVals[n] = expectedUnitScore - heardUnitScore
    #Sum the values
    output = np.sum(scoreVals)
    #Divide them by the number of units in a sequnce
    output = output/(N)
    return output

#Find the second lowest index in a list
def secondLowestIndex(input):
    a = list(input)
    b = list(input)

    b.pop(b.index(min(b)))

    valueToGet = min(b)

    output = a.index(valueToGet)

    return output












        




