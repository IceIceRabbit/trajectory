import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la

#period of testing in seconds
time = 10
goal = np.array([2,200])


x1 = []
y1 = []
x2 = []
y2 = []

#initial positions of both enities
x1_in = 2
y1_in = 0
x2_in = 2
y2_in = 20

r1 = 1
r2 = 0.5
epsilon = 0
v1 = 1
v2 = 0 # assuming a static obstacle
a1 = 0 # constant velocity for both enities
a2 = 0
theta1 = math.pi/2 #assuming straight line trajectory
theta2 = math.pi/2

action_space = [-math.pi/4,-math.pi/6,0,math.pi/6,math.pi/4] #set of possible action defined for [FOR TESTING]

def distance_from_collision(A1,A2):
    la.norm(A1 - A2)
    
def col_reward(X1,X2,Y1,Y2): #old reward function
    reward = (1/(la.norm(np.array([X1,Y1]) - np.array([X2,Y2])))) * (math.pow((X1 - X2),2) + math.pow((X1 - X2),2) - math.pow((r1 + r2 + epsilon),2))
    return reward

def distance_to_goal(x1,y1):
    dist = la.norm(np.array([x1,y1]) - goal)
    return dist

def reward(x1,y1,dmin):
    if(dmin < 0):
        reward = -0.25
    elif(dmin < 2):
        reward = -0.1 -0.5*dmin
    elif(distance_to_goal(x1,y1) == 0):
        reward = 1
    else:
        reward = 0.01

    return reward

def step(action,timestep):
    x1.append(x1_in + ((v1/10)*timestep + a1*(math.pow(timestep,2)))*math.cos(action_space[action])) #agent movement according action provided
    y1.append(y1_in + ((v1/10)*timestep + a1*(math.pow(timestep,2)))*math.sin(action_space[action]))
    x2.append(x2_in + ((v2/10)*timestep + a2*(math.pow(timestep,2)))*math.cos(theta2))
    y2.append(y2_in + ((v2/10)*timestep + a2*(math.pow(timestep,2)))*math.sin(theta2))
    timestep += 1
    done = False
    if distance_to_goal(x1[timestep-1],y1[timestep-1]) == 0:
        done = True
    return [x1[timestep-1],y1[timestep - 1]] , reward( x1[timestep-1] , y1[timestep-1] , distance_from_collision(np.array([x1[timestep-1],y1[timestep-1]]),np.array([x2[timestep-1],y2[timestep-1]]))), done,timestep

def plot():
    plt.plot(x1,y1)
    plt.plot(x2,y2,'ro')
    plt.show()
