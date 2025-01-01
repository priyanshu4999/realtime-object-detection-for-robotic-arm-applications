
from detecto import core, utils, visualize
import time
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
import cv2
print(torch.cuda.is_available())
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.get_device_name(0))
model = core.Model()

#labels, boxes, scores = model.predict_top(image)
#visualize.show_labeled_image(image, boxes, labels)

dataset = core.Dataset('ducky/')
print(len(dataset))
datatest = core.Dataset('mixduckball/mix/')
model = core.Model(['rubberduck','ball'] , device = "cuda:0")

#losses = model.fit(dataset,verbose=True,epochs = 50)


#model.save('model_weights_rubberduck.pth.pth')

# ... Later ...

model = core.Model.load('xmlread_mixduckball.pth', ['rubberduck','ball'])
image = utils.read_image('ducky/ducky55.jpg')
print(image.shape)
image1 = cv2.resize( image , (144 , 144), interpolation = cv2.INTER_AREA)
def prediction(image):
    return model.predict_top(image)
a = time.time()
print(0)

#visualize.show_labeled_image(image, boxes, labels)
for i in range(10) :
    #labels, boxes, scores = model.predict_top(image)
    a1 = time.time()
    image, targets = datatest[i]
    #labels1, boxes1, scores1 = model.predict_top(image1)
    labels, boxes, scores = prediction(image)
    b1 = time.time()
    print(b1 - a1)
    visualize.show_labeled_image(image, boxes, labels)
#visualize.show_labeled_image(image1, boxes1, labels1)
#print(labels) 

#print(boxes1*360/144)
#print(boxes1)
#print(scores)
b = time.time()

print(b-a)
#plt.plot(losses)
plt.show()
