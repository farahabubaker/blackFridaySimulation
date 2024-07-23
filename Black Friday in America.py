# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:34:15 2022

@author: farah
"""

import numpy as np  
import random  
import matplotlib.pyplot as plt

max_col=21
max_row=20
max_agents=200
num_items=10  #tv, phone, computer for now, will increase
  
# 2-d matrices of the store
"""
1. entrances are hardcoded. For now, only one entrance with one spot
on the grid
2. item_stock shows how much stock is left for the item when agents
come grab item
3. For now: 1 = tv, 2 = phone. Item number - 1 = item_stock check
4. For now there are no registers. Agent walks in, grabs items,
walks out
"""

unhappy_agent = np.zeros((20), np.uint8)
item_stock = np.zeros((num_items), np.uint8)
item_weight = np.zeros((num_items), dtype = float)
item_weight_agent = np.zeros((num_items), dtype = float)
store = np.zeros((max_row,max_col), np.int8) 
agent_loc = np.zeros((max_row,max_col), np.int8)

store[2][2] = 1
store[10][2] = 2
store[10][5] = 3
store[7][2] = 4
store[7][10] = 5
store[18][2] = 6
store[10][10] = 7
store[18][7] = 8
store[18][12] = 9
store[18][15] = 10
# splitting items and creating more locations for popular items
# store[2][6] = 11
# store[2][10] = 12
# store[13][5] = 13
# store[2][14] = 14


agent_loc[2][2] = -1
agent_loc[10][2] = -1
agent_loc[10][5] = -1
agent_loc[7][2] = -1
agent_loc[7][10] = -1
agent_loc[18][2] = -1
agent_loc[10][10] = -1
agent_loc[18][7] = -1
agent_loc[18][12] = -1
agent_loc[18][15] = -1
# splitting items and creating more locations for popular items
# agent_loc[2][6] = -1
# agent_loc[2][10] = -1
# agent_loc[13][5] = -1
# agent_loc[2][14] = -1


item_locations=np.array([[2,2], [10,2], [10,5], [7,2], [7,10], [18,2], [10,10], [18,7], [18,12], [18,15]])# [2,6],[2,10], [13,5], [2,14]])
item_stock[0] = 100
item_stock[1] = 50
item_stock[2] = 25
item_stock[3] = 50
item_stock[4] = 100
item_stock[5] = 100
item_stock[6] = 100
item_stock[7] = 100
item_stock[8] = 100
item_stock[9] = 100
# splitting items and creating more locations for popular items
# item_stock[10] = 50
# item_stock[11] = 50
# item_stock[12] = 50
# item_stock[13] = 25

# Actual item weight - represents the item popularity or 'discounted desirable items'
# only uncommented when BF = true
item_weight[0] = 1
item_weight[1] = 2
item_weight[2] = 2
item_weight[3] = 2
item_weight[4] = 1.5
item_weight[5] = 1.7
item_weight[6] = 1.5
item_weight[7] = 1.5
item_weight[8] = 1.9
item_weight[9] = 1.8

# item_weight[10] = 2
# item_weight[11] = 2
# item_weight[12] = 2
# item_weight[13] = 2

entranceWest=[20,2]
entranceSouth = [20,18]
exits=[20,10]


class Agent:
  def __init__(self, entrance, exits, item_weight_agent):
    self.entrance = entrance[::-1].copy()
    self.weight = item_weight_agent.copy()
    self.exits = exits[::-1].copy()
    self.location = entrance[::-1].copy()
    self.has_exit = False 
    self.not_happy = False
    self.num_items_retrieved = 0


#############################################################################
# Agents are created with predetermined item preferences
def define_agents():
    agents = np.ndarray((max_agents), dtype = object)
    for j in range(max_agents):
        for i in range (num_items):
            item_weight_agent[i] = round(random.random(), 2)
        dice=random.randint(0,1)
        if dice == 0:
            agents[j] = Agent(entranceWest, exits, item_weight_agent)
        else:
            agents[j] = Agent(entranceSouth, exits, item_weight_agent)
    return agents


#################################
#
# Dijkstras Algorithm
#
#################################
def dijkstra(loc_val):
    visited =np.zeros((max_row,max_col),dtype=int)
    distance = np.ones((max_row,max_col),dtype=int)
    distance = -1 * distance
    
    #need to add entrance here, right now simplified one entrance
    item_loc_row = loc_val[0]
    item_loc_col = loc_val[1]
    distance[item_loc_row,item_loc_col]=0
    for t in range (100):
     for x in range (max_row):
        for y in range (max_col):
            if (visited[x,y]==0) and (store[x,y]==0) and (distance[x,y]!=-1):
                if x<(max_row-1) :
                 if ((distance[x+1,y]>(distance[x,y]+1)) or (distance[x+1,y]==-1)) and store[x+1,y]==0:
                   distance[x+1,y]= distance[x,y]+1
                if y<(max_row-1):   
                 if ((distance[x,y+1]>(distance[x,y]+1))or (distance[x,y+1]==-1)) and store[x,y+1]==0:
                   distance[x,y+1]= distance[x,y]+1
                if x>0 :
                 if ((distance[x-1,y]>(distance[x,y]+1))or (distance[x-1,y]==-1))and store[x-1,y]==0:
                    distance[x-1,y]= distance[x,y]+1
                if y>0:
                 if ((distance[x,y-1]>(distance[x,y]+1))or (distance[x,y-1]==-1))and store[x,y-1]==0:
                   distance[x,y-1]= distance[x,y]+1
                visited[x,y]=1 
    
    #plt.figure(figsize = (16,4))
    #im=plt.imshow(distance,cmap="gist_ncar") 
    ## see https://matplotlib.org/examples/color/colormaps_reference.html
    #plt.colorbar(im)
    #plt.show()
     
    #print (distance)
    return distance

#################################

def find_greater_val_randweights(item_weight_agent, find_path):

    # This function compares the agent's preference to an item (item weight) with the distance
    # it would take to get to the item
    great_val = np.zeros((num_items), dtype = float)
    
    for i in range (0,num_items):
        item_loc_row = item_locations[i][0]
        item_loc_col = item_locations[i][1]
        
        val=9999
        if(find_path[item_loc_row+1][item_loc_col]) < val and (find_path[item_loc_row+1][item_loc_col]) > 0:
            val = find_path[item_loc_row+1][item_loc_col]
            
        elif(find_path[item_loc_row-1][item_loc_col]) < val and (find_path[item_loc_row-1][item_loc_col]) > 0:
            val = find_path[item_loc_row-1][item_loc_col]
            
        elif(find_path[item_loc_row][item_loc_col+1]) < val and (find_path[item_loc_row][item_loc_col+1]) > 0:
            val = find_path[item_loc_row][item_loc_col+1]

        elif(find_path[item_loc_row][item_loc_col-1]) < val and (find_path[item_loc_row][item_loc_col-1]) > 0:
            val = find_path[item_loc_row][item_loc_col-1]
        
        else:
            val = 0
        
        great_val[i] = val * item_weight_agent[i] * item_weight[i]

    return great_val


############################################################################################
def bf_w_randweights(agents):
    time = 0
    for t in range(1,21):
        agent=0
        while agent<max_agents:
            time = time+1            
            if agents[agent].has_exit == False:
                find_path = dijkstra(agents[agent].location)   
                great_val = find_greater_val_randweights(agents[agent].weight, find_path)    
                result = np.where(great_val == np.amax(great_val))
            
                # If the item is in stock and the agent has less items in their basket
                # than in the store, then the agent's location will be checked to see
                # if it can go to the desired item
                if item_stock[result[0][0]] > 0:
                    item_stock[result[0]] = item_stock[result[0]]-1                
                    if agents[agent].num_items_retrieved == num_items:
                        agent_loc[agents[agent].location[1]][agents[agent].location[0]] = 0
                        agents[agent].location = agents[agent].exits                      
                        agents[agent].has_exit = True
                        #print("Agent has checked out")
                    else:
                        agents[agent].num_items_retrieved = agents[agent].num_items_retrieved + 1
                        loc_val = item_locations[result[0]]
                        check_location_for_agent(loc_val, agents[agent], agent, t)                  
                else:
                    #if the desired item is not in stock, then the agent has the option
                    # of continuting shopping or leaving the store
                    agents[agent].not_happy = True
                    unhappy_agent[t] = unhappy_agent[t]+1
                    dice=random.randint(0,1)
                    if dice == 0:
                        agent_loc[agents[agent].location[0]][agents[agent].location[1]] = 0
                        agents[agent].location = agents[agent].exits                      
                        agents[agent].has_exit = True   
                        
            if (time)%max_agents == 0:
                plot_store(t)
            agent=agent+1
        
        
"""
            
    "Observers indicated that 60% of the shoppers appeared to have a specific
product in mind, particularly electronic media items, such as DVDs, DVD players, televisions, digital
cameras, digital photo frames, Wii, Playstation3, and X-box games."
            
            
        """
############################################################

def check_location_for_agent(loc_val, agent, which_agent, t):
    item_loc_row = loc_val[0][0]
    item_loc_col = loc_val[0][1]
        
    if agent_loc[item_loc_row][item_loc_col] == -1:        
        for i in range(item_loc_col-1, item_loc_col+2): 
            for j in range(item_loc_row-1, item_loc_row+2):
                if agent_loc[j][i] == 0:                  
                    agent_loc[agent.location[0]][agent.location[1]] = 0
                    agent_loc[j][i] = which_agent+1
                    agent.location[0] = j
                    agent.location[1] = i  
                    agent.not_happy = False
                    return
         # If the agent is unable to be near the desired item and is forced to be further
         # away then the agent becomes 'unhappy'
        for i in range(item_loc_col-2, item_loc_col):
            for j in range(item_loc_row-2, item_loc_row):
                if agent_loc[j][i] == 0:                  
                    agent_loc[agent.location[0]][agent.location[1]] = 0
                    agent_loc[j][i] = which_agent+1
                    agent.location[0] = j
                    agent.location[1] = i  
                    agent.not_happy = True
                    unhappy_agent[t] = unhappy_agent[t]+1
                    print("agent unhappy")
                    return                          
    else:
        #print(item_loc_row, item_loc_col, agent_loc[item_loc_row][item_loc_col])     
        print("agent stuck")
                
    

#################################################################################

def plot_store(time): 
    plt.figure() 
    plt.axis([0, max_col, max_row, 0]) 
    
    # put entrance/exit plot in loop in case multiple entrances and exits
    plt.title("Black Friday in a Store")
    #plt.title("Agents in a Store not on Black Friday")
    plt.plot(entranceWest[1], entranceWest[0], 'rs')
    plt.plot(entranceSouth[1], entranceSouth[0], 'rs')
    plt.plot(exits[1], exits[0], 'bs')
    plt.text(-0.5,-0.5, "Time = " + str(time) ,ha='center',weight='bold')
    
    for i in range (0,num_items):
        if item_stock[i] == 0:
            plt.plot(item_locations[i][0],item_locations[i][1], 'b^')
        else:
            plt.plot(item_locations[i][0],item_locations[i][1], 'g^')
        
    for j in range (0, max_agents):   
        if agents[j].not_happy == True:
            plt.plot(agents[j].location[0], agents[j].location[1], 'r*')
        else:
            plt.plot(agents[j].location[0], agents[j].location[1], 'k*')

#################################################################################
def chart_unhappy_agents(): 
    fig = plt.figure() 
    ax = fig.add_axes([0,0,1,1])
    num_bins = np.arange(20)
    # the histogram of the data
    ax.bar(num_bins, unhappy_agent, color='b')
    ax.set_xticks(np.arange(1, 21, 1))
    
    ax.set_xlabel('Time')
    ax.set_ylabel('# of Unhappy Agents')
    ax.set_title(r'Unhappy Agents during Black Friday')
    plt.show()
    
    print(unhappy_agent)


###################################################################################
# This is the main functioning of the program
agents = define_agents()
plot_store(0)
bf_w_randweights(agents)
chart_unhappy_agents()
