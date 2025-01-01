import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
#data_raw = np.genfromtxt('position2_99.csv', delimiter=',')


####################### remember to test and train on observations from same dimension (360 , 360 in this case)
data_raw = np.genfromtxt('position_final5.csv', delimiter=',')
mean = np.mean(data_raw , axis = 0)
std= np.std(data_raw , axis = 0)
mean_ip = mean[:4]
std_ip = std[:4]
mean_op = mean[4:]
std_op = std[4:]
data_whole = (data_raw - mean)/ std

test_ratio = 0.1#0.05
test_size = int(test_ratio*len(data_whole))
test_index  = np.random.choice(len(data_whole),test_size, replace = False )
data_test = data_whole[test_index]
data_train_preval = np.delete(data_whole , test_index , axis = 0)

val_ratio = 0.1
val_size = int(val_ratio*len(data_train_preval))
val_index  = np.random.choice(len(data_train_preval),val_size, replace = False )
data_val = data_train_preval[val_index]
data_train = np.delete(data_train_preval , val_index , axis = 0)




#ip_data  = (data_train[ :, [0,1]] + data_train[ :, [2,3]])/2
#ip_data_test  = (data_test[ :, [0,1]] + data_test[ :, [2,3]])/2
#ip_data_val  = (data_val[ :, [0,1]] + data_val[ :, [2,3]])/2
ip_data  = data_train[ :, [0,1,2,3]]
ip_data_test  = data_test[ :, [0,1,2,3]]
ip_data_val  = data_val[ :, [0,1,2,3]]

inp_shape = ip_data.shape[1]

op_data = data_train[ :,4: ]
op_data_test = data_test[ :,4: ]
op_data_val = data_val[ :,4: ]

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=inp_shape ),
   tf.keras.layers.Dense(6, activation='relu',kernel_initializer='GlorotNormal'),
   tf.keras.layers.Dense(10, activation= 'relu',kernel_initializer='GlorotNormal'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(6, activation='relu',kernel_initializer='GlorotNormal'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])

model1 = tf.keras.models.Sequential([
   tf.keras.layers.Input(shape=inp_shape ),
  #tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(32, activation='relu',kernel_initializer='GlorotNormal'),
   tf.keras.layers.Dense(16, activation='relu',kernel_initializer='GlorotNormal'),
   tf.keras.layers.Dense(8, activation= 'relu',kernel_initializer='GlorotNormal'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4, activation='relu',kernel_initializer='GlorotNormal'),

  tf.keras.layers.Dense(2 )
])

model2 = tf.keras.models.Sequential([
   tf.keras.layers.Input(shape=inp_shape),
   tf.keras.layers.Dense(6, activation='relu',kernel_initializer='GlorotNormal'),
   tf.keras.layers.Dense(10, activation= 'relu',kernel_initializer='GlorotNormal'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(6, activation='relu',kernel_initializer='GlorotNormal'),

  tf.keras.layers.Dense(2 )
])

optim = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')

model.compile(optimizer=optim,#'adam',
              loss='mse',
              metrics=['accuracy'])

model1.compile(optimizer=optim,#'adam',
              loss='huber_loss',
              metrics=['accuracy'])

model2.compile(optimizer=optim,#'adam',
              loss='huber_loss',
              metrics=['accuracy'])

epoch = 800


#model.fit(ip_data , op_data , epochs = epoch,validation_data=(ip_data_val, op_data_val),batch_size=128)
#model1.fit(ip_data , op_data , epochs = epoch,validation_data=(ip_data_val, op_data_val),batch_size=128)
history = model2.fit(ip_data , op_data , epochs = epoch,validation_data=(ip_data_val, op_data_val),batch_size=128,verbose=0)
#model.save('model_weightspos_1.pth')
#model2.save('model_weightspos2_1.pth')
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

n = 2
#checkpoint_model = f"position/checkpoint/modelcp45{n}.ckpt"#f"position/checkpoint/modelcp-best{n=2}.ckpt"
#checkpoint_model1= f"position/checkpoint/model2cp45{n}.ckpt"#f"position/checkpoint/model1cp-best{n}.ckpt"
#checkpoint_model2= f"position/checkpoint/model3cp45{n}.ckpt" #f"position/checkpoint/model2cp-best{n}.ckpt"
#checkpoint_model = f"position/checkpoint/final_m0_position{n}.ckpt"
#checkpoint_model1= f"position/checkpoint/final_m1_position{n}.ckpt"
checkpoint_model2= f"position/checkpoint/final_m2_position{n}.ckpt"# n = 2 for best results saved

#model1.load_weights(checkpoint_model1)
model2.load_weights(checkpoint_model2)
#model.save_weights(checkpoint_model.format(epoch=0))
#model1.save_weights(checkpoint_model1.format(epoch=0))
#model2.fit(ip_data , op_data , epochs = epoch,validation_data=(ip_data_val, op_data_val),batch_size=128)
#model2.save_weights(checkpoint_model2.format(epoch=0))

plot_data = []
for i in range(len(data_test)):
   #predict = np.around(mean_op + std_op*model.predict(ip_data_test[i].reshape(1,len(ip_data_test[0]))),4)
   #predict1 = np.around(mean_op + std_op*model1.predict(ip_data_test[i].reshape(1,len(ip_data_test[0]))),4)
   predict2 = np.around(mean_op + std_op*model2.predict(ip_data_test[i].reshape(1,len(ip_data_test[0]))),4)
   true = np.around(mean_op+ std_op*op_data_test[i] , 4)
   plot_data.append(predict2 - true)
   #final = np.around((predict2 +  predict1 )/ 2 , 4)
   #print(f"  0: {predict2 - true}  , 2:{predict2 - true}    , final : { predict2}" )
#print(mean_op )
#print(std_op )
meanip = np.array([135.5      ,  128.17151767 ,163.29417879 ,155.82536383])
stdip = np.array([29.33388163 ,21.8132477 , 29.25697577 ,22.47152648])
meanop = np.array([ 0.55173493 ,-0.09533012])
stdop = np.array([0.06785992 ,0.06972869])

plotdata = np.array(plot_data)
finaldata = plotdata.reshape(-1,2).squeeze()
#print(finaldata)
x = finaldata.T[0]
y = finaldata.T[1]
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n_bins = 100
# We can set the number of bins with the `bins` kwarg
plt.xlim([-0.08, 0.08])
axs[1].hist(x, bins=n_bins)
plt.xlim([-0.08, 0.08])
axs[0].hist(y, bins=n_bins)
plt.xlim([-0.08, 0.08])

plt.show()

fig, ax = plt.subplots(tight_layout=True)
plt.xlim([-0.08, 0.08])
plt.ylim([-0.08, 0.08])
hist = ax.hist2d(x, y , bins = 100)

plt.show()
