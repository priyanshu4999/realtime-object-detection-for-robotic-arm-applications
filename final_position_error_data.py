import pybullet as p
import time
import datetime
import pybullet_data
import os
import numpy as np
import copy
import math
import random
import torch
from kuka_original import Kuka
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import msvcrt
from multiprocessing import Process
import deepq2
from deepq2 import MyModel
from deepq2 import DQN

import cv2
import threading
from detecto import core, utils, visualize
global model
global pos
global counter

model = core.Model()
model = core.Model(['rubberduck'])
model = core.Model.load('xmlread_mixduckball.pth', ['rubberduck',"ball"])
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

def prediction(image):
    return model.predict_top(image)

##############################
from buffer import ReplayBuffer
x = datetime.datetime.now()
writer = tf.summary.create_file_writer(f"tmp/log/{x.day}_{x.month}_{x.year}  {x.hour }.{x.minute }")
#################
cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
############################## Kuka class predefined
############ ### crentre set at base of kuka
############################################## in deepq1 set n = 1
def fun():
  j = 0
  while 1:
    print(f"dnsjfnjsbdkjfjkbsdjfbjksdbfkjjsdbfjbsdjbfkjbsdjkbfjk         {++j}")
    if j==100:
      return "sfsldfjhdbfjbdsjbfbdsjkbfkbsdbfjkbdskfbjdsbfjkbsjd" 

def basicEnv():
  kuka = Kuka()
  shift = [1.3, -0.7, 0.75] ## when camera feed viewd from top
  shift[0] = shift[0] - 0.7 # forward backward
  shift[1] = shift[1] + 0.6 # left right
  shift[2] = shift[2] - 0.15 # up down
  plane = p.loadURDF("plane.urdf"       , 0.000000 - shift[0],  0.000000 - shift[1], 0.000000 - shift[2], 0.000000, 0.000000, 0.000000, 1.000000)
  table = p.loadURDF("table/table.urdf" , 0.900000 - shift[0], -0.200000 - shift[1], 0.000000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107)
  tray =p.loadURDF("tray/tray.urdf"   , 1.070000 - shift[0],  -0.200000 - shift[1], 0.52500000 - shift[2], 0.000000, 0.000000, 0.000000, -1.000000)
  objectList = []
  return kuka , plane , table , tray,objectList,shift

def reset(shift,kuka,obj_plane,obj_tray,obj_table,*obj):
  #p.resetSimulation()
  kuka = Kuka()
  kuka.applyAction([0,0,0,1.57/2,0])
  kukaBasePosAndOrn = [-0.100000, 0.000000, 0.070000 , 0.000000, 0.000000, 0.000000, 1.000000]
  gravityAbs = 9.810000

  obj_plane = p.loadURDF("plane.urdf"       , 0.000000 - shift[0],  0.000000 - shift[1], 0.000000 - shift[2], 0.000000, 0.000000, 0.000000, 1.000000)

  obj_table = p.loadURDF("table/table.urdf" , 0.900000 - shift[0], -0.200000 - shift[1], 0.000000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107)

  obj_tray = p.loadURDF("tray/tray.urdf"   , 1.070000 - shift[0],  -0.200000 - shift[1], 0.52500000 - shift[2], 0.000000, 0.000000, 0.000000, -1.000000)
  a = 0.2*(np.random.rand(3)-0.5)
  b = 0.2*(np.random.rand(4)-0.5)
  scale = 0.8 + np.random.rand(7)/2
  obj = []
  name = np.random.choice(["duck_vhacd.urdf", "sphere_small.urdf"],1)
  #-1
  #obj.append(p.loadURDF("duck_vhacd.urdf"  , 1.300000 - shift[0],  0.300000 - shift[1], 0.900000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107))
  #obj.append(p.loadURDF("duck_vhacd.urdf"  , 1.160000 + a[0]- shift[0], -0.200000 + a[1] - shift[1], 0.630000 - shift[2], b[0], b[1],b[2],b[3]))
  #obj.append(p.loadURDF("duck_vhacd.urdf"  , 1.160000 + a[0]- shift[0], -0.200000 + a[1] - shift[1], 0.630000 - shift[2],0.000000, 0.000000, 0.707107, 0.707107))
  obj.append(p.loadURDF(str(name[0])  ,basePosition = [ 1.150000 + a[0]- shift[0], -0.200000 + b[1] - shift[1], 0.650000 - shift[2]], baseOrientation = [b[3], 0.000000, 0.707107, 0.707107] ,globalScaling = scale[1]))
  #obj.append(p.loadURDF("sphere_small.urdf"  ,basePosition = [ 1.150000 + a[0]- shift[0], -0.200000 + b[1] - shift[1], 0.650000 - shift[2]], baseOrientation = [b[3], 0.000000, 0.707107, 0.707107] ,globalScaling = scale[1]))
  #obj.append(p.loadURDF("duck_vhacd.urdf"  ,basePosition = [ 1.150000 + b[0]- shift[0], -0.200000 + a[1] - shift[1], 0.650000 - shift[2]], baseOrientation = [0.000000, 0.000000, 0.707107, 0.707107] ,globalScaling = scale[2]))
  #obj.append(p.loadURDF("duck_vhacd.urdf"  ,basePosition = [ 1.150000 + (a[1]+b[0]/2)- shift[0], -0.200000 + b[1] - shift[1], 0.650000 - shift[2]], baseOrientation = [0.000000, 0.000000, 0.707107, 0.707107] ,globalScaling = scale[4]))
  #obj.append(p.loadURDF("sphere_small.urdf"  ,basePosition = [ 1.150000 + b[0]- shift[0], -0.200000 + (a[1]+a[0]/2) - shift[1], 0.650000 - shift[2]], baseOrientation = [0.000000, 0.000000, 0.707107, 0.707107] ,globalScaling = scale[5]))
  #jenga.urdf"  , 1.160000 + a[0]- shift[0], -0.200000 + a[1] - shift[1], 0.6200000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107))
  #obj.append(p.loadURDF("sphere_small.urdf"  , 1.160000 + a[0]- shift[0], -0.200000 + a[1] - shift[1], 0.6200000 - shift[2], b[0], b[1],b[2],b[3]))
  #obj.append(p.loadURDF("teddy_vhacd.urdf"  , 1.050000 + a[0]- shift[0], -0.200000 + a[1] - shift[1], 0.650000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107)) 
  #obj.append(p.loadURDF("jenga/jenga.urdf" , 1.300000 - shift[0], -0.700000 - shift[1], 1.750000 - shift[2], 0.000000, 0.707107, 0.000000, 0.707107))
  #obj.append(p.loadURDF("jenga/jenga.urdf" , 1.200000 - shift[0], -0.700000 - shift[1], 0.750000 - shift[2], 0.000000, 0.707107, 0.000000, 0.000000))
  #obj.append(p.loadURDF("jenga/jenga.urdf" , 1.100000 - shift[0], -0.700000 - shift[1], 0.750000 - shift[2], 0.000000, 0.707107, 0.000000, 0.707107))
  #obj.append(p.loadURDF("jenga/jenga.urdf" , 1.000000 - shift[0], -0.700000 - shift[1], 0.750000 - shift[2], 0.000000, 0.707107, 0.000000, 0.707107))
  #obj.append(p.loadURDF("jenga/jenga.urdf" , 0.900000 - shift[0], -0.700000 - shift[1], 0.750000 - shift[2], 0.000000, 0.707107, 0.000000, 0.707107))
  #obj.append(p.loadURDF("jenga/jenga.urdf" , 0.800000 - shift[0], -0.700000 - shift[1], 0.750000 - shift[2], 0.000000, 0.000000, 0.000000, 1.000000))
  #obj.append(p.loadURDF("teddy_vhacd.urdf" , 1.050000 - shift[0], -0.500000 - shift[1], 0.700000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107))
  #obj.append(p.loadURDF("cube_small.urdf"  , 1.050000 - shift[0], -0.500000 - shift[1], 0.700000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107))
  #obj.append(p.loadURDF("sphere_small.urdf", 0.950000 - shift[0], -0.500000 - shift[1], 0.700000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107))
  #obj.append(p.loadURDF("duck_vhacd.urdf"  , 0.950000 - shift[0], -0.400000 - shift[1], 0.900000 - shift[2], 0.000000, 0.000000, 0.707107, 0.707107))
  #obj.append(p.loadURDF("lego/lego.urdf"   , 1.000000 - shift[0], -0.500000 - shift[1], 0.900000 - shift[2], 0.000000, 0.000000, 0.000000, 1.000000))
  #obj.append(p.loadURDF("lego/lego.urdf"   , 1.000000 - shift[0], -0.500000 - shift[1], 0.700000 - shift[2], 0.000000, 0.000000, 0.000000, 1.000000))
  #obj.append(p.loadURDF("lego/lego.urdf"   , 1.000000 - shift[0], -0.500000 - shift[1], 0.800000 - shift[2], 0.000000, 0.000000, 0.000000, 1.000000))
  p.setGravity(0, 0, -gravityAbs)
  ref_time = time.time()
  state = distanceBeforeAction = distancefunc(kuka.kukaUid , obj[0],"Vect",1 ).copy()
  time.sleep(1)
  return state,kuka,obj,scale


############## camera
def imageshape():
  width = 360
  height = 360
  return width , height
def cameraLocation(UniqueId , link , campos , shift , fixed):
  #shift == [0,0,0] for no shift
  #campos == fixed position of camera without any shift , final pos = campos - shift
  #-1 for base link
  if fixed == True:
    viewMat= p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0.55,0.00000,0.0000] , yaw = 135.0 , pitch = -50.00 , distance = 1.0 , roll = 0 , upAxisIndex = 2) ##0.1 0 0  ,  90 , -40 , 1.42
    #viewMat = p.computeViewMatrix(cameraEyePosition=[campos[0] - shift[0], campos[1] - shift[1], campos[2] - shift[2]]
                               # ,cameraTargetPosition=[campos[0] - shift[0], campos[1] - shift[1], -100 - shift[2]]
                                #,cameraUpVector=[1, 0, 0]
                                # )
  else: ### change projectionMatrix fov to 60-90 degree
    wristCam = p.getLinkState(UniqueId , link)[0]
    viewMat = p.computeViewMatrix(cameraEyePosition=wristCam
                                ,cameraTargetPosition=[wristCam[0], wristCam[1] , -100 ]
                                ,cameraUpVector=[1, 0, 0]
                                 )
  return viewMat

################# Actions 

def possibleactions(typeOfAction):
  action = [0,0,0,0,0]
  dv = 0.008#0.0015
  if typeOfAction =="stop":
    action = [0,0,0,1.57/5,0]
  if typeOfAction == "left":
    action = [0,dv,-dv,0,1.57/5]
  if typeOfAction == "right":
    action = [0,-dv,-dv,0,1.57/5]
  if typeOfAction == "up":
    action = [0,0,dv,0,0]
  if typeOfAction == "down":
    action = [0,0,-dv,0,1.57/5]
  if typeOfAction == "forward":
    action = [dv,0,-dv,0,1.57/5]
  if typeOfAction == "backward":
    action = [-dv,0,-dv,0,1.57/5]
  if typeOfAction == "open":
    action = [0,0,0,0,1.57/5]
  #if typeOfAction == "rotate":
    #action = [0,0,0,1.57/20,0]
  return action


def actionfunc(index):
  actionList = ["stop","left" , "right" , "down" , "forward" , "backward"  , "up", "open"]
  if index<0 or index>(len(actionList)-1):
    index = 0
  return possibleactions(actionList[index])

def randomNum():
  return np.random.choice([0,1,2,3,4,5,6,7,8]) 
############
projectionMatrix = p.computeProjectionMatrixFOV(fov=45.0,aspect=1.0,nearVal=0.1,farVal=1.6)
k = 0
###########

#actionList = ["stop","left" , "right" , "up" , "down" , "forward" , "backward"  , "open",'rotate']
actionList = ["stop","left" , "right"  , "down" , "forward" , "backward"  , "up", "open"]
############
def distancefunc(uid1 , uid2,typeOfdistance,ifkuka):
  endEff = 11
  
  if np.random.rand(1)>0.7:
    error = np.random.normal(0,0.02,3)
  else:
    error  = error = np.random.normal(0,0.001,3)
  #print(error)
  if ifkuka == 1:
    distanceEucl = np.round(abs(p.getClosestPoints(uid1 , uid2,10,endEff )[0][8]), decimals=4)
    distanceVect = np.round(np.array(p.getLinkState(uid1 , endEff )[0]) - np.array(p.getBasePositionAndOrientation(uid2)[0]), decimals=4) + error
    distanceManh =  abs(distanceVect[0] ) + abs(distanceVect[1] ) #+ abs(distanceVect[2] )
    distanceEucl2 =  np.sqrt(abs(distanceVect[0] )**2 + abs(distanceVect[1] ) **2)
    distance = distanceManh.copy()
  else:
    distanceEucl = np.round(abs(p.getClosestPoints(uid1 , uid2,10)[0][8]), decimals=4)
    distanceVect = np.round(np.array(p.getBasePositionAndOrientation(uid1)[0] - np.array(p.getBasePositionAndOrientation(uid2)[0])), decimals=4) 
    distanceManh =  abs(distanceVect[0] ) + abs(distanceVect[1] ) #+ abs(distanceVect[2] )
    distanceEucl2 =  np.sqrt(abs(distanceVect[0] )**2 + abs(distanceVect[1] ) **2)
    distance = distanceManh.copy()
  if typeOfdistance=="Vect":
    return  distanceVect
  if typeOfdistance=="Manh":
    return distanceManh
  if typeOfdistance=="Eucl2":
    return distanceEucl2 
  else:
    return distanceEucl
  
def Env(kuka,possibleAction,obj,j):
  viewMatrix = cameraLocation(kuka.kukaUid , 7 , [1.00,-0.20 , 2.20] , shift , True)
  distanceBeforeAction = distancefunc(kuka.kukaUid , obj[0],"Vect",1 ).copy()
  d1 = distancefunc(kuka.kukaUid , obj[0],"Eucl2",1 ).copy()
  kuka.applyAction(possibleAction)
  __, _, rgbImg, depthImg, segImg = p.getCameraImage(width=imageshape()[0], height=imageshape()[1], viewMatrix=viewMatrix,projectionMatrix=projectionMatrix) ##camera feed
  state = np.array(rgbImg)
  #state_tensor = tf.convert_to_tensor(state)
  #state_tensor = tf.expand_dims(state_tensor, 0)
  p.setRealTimeSimulation(1)
  distanceAfterAction = distancefunc(kuka.kukaUid , obj[0] ,"Vect",1)
  d2 = distancefunc(kuka.kukaUid , obj[0],"Eucl2" ,1).copy()
  distance = np.absolute(np.round(distanceAfterAction - distanceBeforeAction,decimals=4))
  #p.getBasePositionAndOrientation()
  obj_table_dist = distancefunc(obj[0] ,obj_table,'Vect',0)[2]
  #print(p.getLinkState(kuka.kukaUid, 11)[0])
  reward = 0
  done = 0
  #print(rgbImg[:,:,:3].shape)
  predict = rgbImg[:,:,:3]
  predict = cv2.cvtColor(rgbImg[:,:,:3], cv2.COLOR_BGR2RGB)
  boxes = np.zeros(4)
  scores = []
  #print(torch.cuda.is_available())
  #if iter%4== 3:
  a1 = time.time()
  labels, box, scores = prediction(predict)
  b1 = time.time()
  #print(b1 - a1)
  #if len(scores)>0:
  boxes = np.squeeze(box.numpy().astype(int))
  #print(f"CONTACT FLAG    {len(p.getContactPoints(bodyA = kuka.kukaUid,bodyB  =  obj[0],linkIndexA = 9))}   {len(p.getContactPoints(bodyA = kuka.kukaUid,bodyB  =  obj[0],linkIndexA = 12))}")
  #if len(boxes)>0:
    #print(boxes)
    #print(scores)

  predict = cv2.rectangle(predict, (boxes[0],boxes[1]),(boxes[2],boxes[3]), (255,0,0), 2)
  if iter%3== 2: 
    cv2.imshow("img_2",predict )
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("lol")
  if d2<d1:
    reward = 0.01 + np.exp(-d2)/1000 + (len(p.getContactPoints(bodyA = kuka.kukaUid,bodyB  =  obj[0],linkIndexA = 13))>0)*np.exp(-d2)/1000 
    #print("lifted")
  elif d2>d1:
    reward = -0.01 
    done = 0
  else:
    reward = 0
  if obj_table_dist>0.175:
    reward =  0.1
    done = 1    
  #data = predict
  data = np.squeeze(np.array([boxes[0],boxes[1],boxes[2],boxes[3] , p.getBasePositionAndOrientation(obj[0])[0][0],p.getBasePositionAndOrientation(obj[0])[0][1]]))
  return distanceBeforeAction  , reward , distanceAfterAction ,done, data
#####################

counter = 2
max_length = 1000000
input_shape =  3
n_actions = 6
hiddenUnits_Sizes = [32,32]

gamma = 0.99
max_xp = 10000
min_xp = 201
batch_size = 32
lr = 0.0001
epsilon = 0.99
min_epsilon = 0.1
decay= 0.9995
update_Q_steps= 1000

losses = list()
reward_s = 0
done = 0
Points  = []
#memory = ReplayBuffer(max_length ,input_shape ,n_actions ) ######## check buffer.py
####################
Q = DQN(input_shape , n_actions ,  hiddenUnits_Sizes , gamma ,max_xp ,min_xp , batch_size , lr)
Q.model.load_weights("tmp/log/checkpoint/cp-best1.ckpt")
#Q_= DQN(input_shape , n_actions ,  hiddenUnits_Sizes , gamma ,max_xp ,min_xp , batch_size , lr)
####################
kuka,obj_plane,obj_tray,obj_table,obj,shift = basicEnv()
state, kuka , obj ,scale= reset(shift , kuka,obj_plane,obj_tray,obj_table,*obj)
initial_pos = kuka.endEffectorPos.copy()
ep_reward = 0
ep = 0
c = 0
d = 0
analysis = []
global completed
#################################
iter = 0
action = 2
while(1):
  
  action = Q.get_action(state, 0)
  state , reward , state_,done ,data = Env(kuka,actionfunc(action),obj,iter)
  x = threading.Thread(target=fun, args=(1,), daemon=True)
  a1 = time.time()
  completed = 0
  #label , box , idf = prediction(data)
  b1 = time.time()
  #print(b1 - a1)
  #print(iter)
  #if iter%8 == 7:
    #cv2.imwrite(f"mix6{iter}.jpg" , data)
    #print(action)
  #if iter%45== 44:
    #d = 1
    #for k in range(50):#50
      #state , reward , state_,done,data = Env(kuka,actionfunc(6),obj,iter+1)
      #print("up")
    #ep_reward = 0
      

  if d:
    ep +=1
    print(ep)
    c = c + done
    analysis.append([c,ep,scale[1]])
      #np.savetxt(f"analysis.csv", analysis, delimiter=",")
    print(f"objective achieved {c} out of {ep} scale {scale[1]} ")
    state,kuka,obj,scale= reset(shift,kuka,obj_plane,obj_tray,obj_table,*obj)
    ep_reward += reward
    d = 0
    

  if iter%3== 2 :
    Points.append(np.squeeze(data.flatten()))
    print(np.squeeze(data.flatten()))
    print(len(np.array(Points)))
  
  iter+=1


  if len(Points)%10== 9:
    np.savetxt(f"position_final4.csv", Points, delimiter=",")

#################
p.disconnect()
