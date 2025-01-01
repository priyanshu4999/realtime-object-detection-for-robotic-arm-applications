
from detecto import core, utils, visualize
import time
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import cv2
import numpy as np
from detecto.visualize import show_labeled_image

Ap = []
p = 0
r = 0
total_objects = 0
def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
model = core.Model()

#labels, boxes, scores = model.predict_top(image)
#visualize.show_labeled_image(image, boxes, labels)

dataset = core.Dataset('mixduckball/')
print(type(dataset))

datatest = core.Dataset('mixduckball/testball/')
val_dataset = core.Dataset('mixduckball/testball/')
#image, targets = datatest[0]
#show_labeled_image(image, targets['boxes'], targets['labels'])

model = core.Model(['rubberduck','ball'] , device = "cuda:0")
loss = (model.fit(dataset,val_dataset,verbose=True,epochs = 30))
plt.plot(loss)
plt.show()
#losses = model.fit(dataset,verbose=True,epochs = 30)
model.save('xmlread_mixduckball_smalldata.pth')
#model.save('model_weights_rubberduck.pth.pth')

# ... Later ...

model = core.Model.load('xmlread_mixduckball_test.pth', ['rubberduck','ball'])
image = utils.read_image('mixduckball/mix639.jpg')
print(image.shape)
image1 = cv2.resize( image , (360, 360), interpolation = cv2.INTER_AREA)
def prediction(image):
    return model.predict_top(image)
a = time.time()
print(0)

#visualize.show_labeled_image(image, boxes, labels)
for i in range(1) :
    #labels, boxes, scores = model.predict_top(image)
    #image = utils.read_image('mixduckball/mix639.jpg')
    image = utils.read_image('mixduckball/mix639.jpg')
    #image, targets = datatest[i]
    a1 = time.time()
    labels, boxes, scores = model.predict(image)
    a = np.array(scores)
    b = np.array(np.where(a>0.9)).flatten()
    
    
    #prediction(image1)
    b1 = time.time()
    #print(b1 - a1)
    #print((scores[:len(b)])) 
    visualize.show_labeled_image(image, boxes[:len(b)], labels[:len(b)])
    #visualize.show_labeled_image(image, targets['boxes'], labels[:len(b)])
    for i in range(len(b)):
        #iou_avg.append(IOU(boxes[i] , targets['boxes'][i]))
        #iou = IOU(boxes[i] , targets['boxes'][0])
       #print(f" IOU is  {iou}")
        #print(len(targets["labels"]))
        total_objects +=1
        #r += int(scores[i]>0.9)
        #p += int(scores[i]>0.9)* (iou>0.8)
        #Ap.append([p ,r])

#print(np.mean(iou_avg))
#visualize.show_labeled_image(image, boxes, labels)

#print(boxes1*360/144)
#print(boxes1)
#print(scores)
#b = time.time()
#print(b-a)


#np.savetxt(f"Resnet-50.csv", losses , delimiter=",")
#AP_ = np.array(Ap)
#AP = AP_.T

r0= [0,1/15 , 2/15 ,3/15 ,4/15 ,5/15 , 6/15 , 7/15 , 8/15 , 9/15, 9/15 , 10/15 ,11/15 , 12/15 , 12/15 , 13/15 , 14/15 , 15/15,1]
p0 = [1,1/1 , 2/2 ,3/3, 4/4 ,5/5 ,6/6 ,7/7 ,8/8 ,9/9, 10/10 ,11/11, 12/12 ,13/13 ,14/14 ,15/15 ,15/16 ,15/17,0]
Ap = np.array([r,p]).T
r = [0,1/15 , 2/15 ,3/15 ,4/15 ,5/15 , 6/15 , 7/15 , 8/15 , 9/15, 9/15 , 10/15 ,11/15 , 12/15 , 12/15 , 13/15 , 14/15 , 14/15,14/15]
p = [1,1/1 , 2/2 ,3/3, 4/4 ,5/5 ,6/6 ,7/7 ,8/8 ,9/9, 10/10 ,11/11, 12/12 ,13/13 ,14/14 ,14/15 ,15/16 ,15/17,0]
Ap = np.array([r,p]).T
r1= [0,1/15 , 2/15 ,3/15 ,4/15 ,5/15 , 6/15 , 7/15 , 8/15 , 9/15, 9/15 , 10/15 ,11/15 , 11/15 , 12/15 , 12/15 , 13/15 , 13/15,13/15]
p1= [1,1/1 , 2/2 ,3/3, 4/4 ,5/5 ,6/6 ,7/7 ,8/8 ,9/9, 10/10 ,11/11, 12/12 ,12/13 ,13/14 ,13/15 ,13/16 ,14/17,0]


plt.plot(r0,p0 , label="mAP 0.963 , IOU>0.8")
plt.plot(r,p ,  label="mAP 0.894 , IOU>0.85")
plt.plot(r1,p1,   label="mAP 0.827 , IOU>0.90")
plt.legend()
plt.show()
