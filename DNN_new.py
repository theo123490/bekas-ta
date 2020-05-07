import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp

data = pd.read_pickle('Feature Data.pk')
#data = data.sort_values(['Diagnosis'],axis=0,ascending=True)
data = data.sample(frac=1)
train_labels = data.loc[:,'Diagnosis'].values
train_labels = train_labels.astype(int)



sort_data = data.sort_values(['Diagnosis'],axis=0,ascending=True)

sort_data_a = sort_data.drop(columns = ['Diagnosis','GLCM_mat','Normalize_Hist'])
sort_data_a = sort_data_a.values.astype(float)


sort_data_b = sort_data.loc[:,'Normalize_Hist'].values
sort_data_b = np.concatenate(sort_data_b)
sort_data_b = np.reshape(sort_data_b,(-1,8,8,8))
sort_data_b = sort_data_b/np.max(sort_data_b)


sort_data_a[:,0] = sort_data_a[:,0]/np.max(sort_data_a[:,0])
sort_data_a[:,1] = sort_data_a[:,1]/np.max(sort_data_a[:,1])
sort_data_a[:,2] = sort_data_a[:,2]/np.max(sort_data_a[:,2])
sort_data_a[:,3] = sort_data_a[:,3]/np.max(sort_data_a[:,3])
sort_data_a[:,4] = sort_data_a[:,4]/np.max(sort_data_a[:,4])
sort_data_a[:,5] = sort_data_a[:,5]/np.max(sort_data_a[:,5])
sort_data_a[:,6] = sort_data_a[:,6]/np.max(sort_data_a[:,6])
sort_data_a[:,7] = sort_data_a[:,7]/np.max(sort_data_a[:,7])

sort_data_a = np.nan_to_num(sort_data_a)
sort_data_b = np.nan_to_num(sort_data_b)
sort_train_labels = sort_data.loc[:,'Diagnosis'].values
sort_train_labels = sort_train_labels.astype(int)
sort_y_binary = keras.utils.to_categorical(sort_train_labels).astype(int)

test_data_n = 200

train_data_a = np.concatenate([sort_data_a[test_data_n:-test_data_n,:]])
train_data_b = np.concatenate([sort_data_b[test_data_n:-test_data_n,:,:,:]])
train_y = np.concatenate([sort_y_binary[test_data_n:-test_data_n,:]])

test_data_a = np.concatenate([sort_data_a[0:test_data_n,:],sort_data_a[-test_data_n:,:]])
test_data_b = np.concatenate([sort_data_b[0:test_data_n,:,:,:],sort_data_b[-test_data_n:,:,:,:]])
test_y = np.concatenate([sort_y_binary[0:test_data_n,:],sort_y_binary[-test_data_n:,:]])





class_names = ['Nevus', 'Melanoma']

color_input = keras.layers.Input((8,8,8))
vector_input = keras.layers.Input((8,)) 

v_dense_layer = keras.layers.Dense(10,activation="softsign")(vector_input)

color_flat_layer = keras.layers.Flatten()(color_input)
color_dense = keras.layers.Dense(100,activation="softsign")(color_flat_layer)


concat_layer= keras.layers.Concatenate()([v_dense_layer, color_dense])
con_out_layer = keras.layers.Dense(400,activation="softplus")(concat_layer)
con_out_layer1 = keras.layers.Dense(300,activation="softsign")(con_out_layer)
con_out_layer2 = keras.layers.Dense(200,activation="softsign")(con_out_layer1)
con_out_layer3 = keras.layers.Dense(200,activation="softsign")(con_out_layer2)
con_out_layer4 = keras.layers.Dense(70,activation="softsign")(con_out_layer3)

output = keras.layers.Dense(2,activation="softplus")(con_out_layer4)

model = keras.models.Model(inputs=[vector_input, color_input], outputs=output)
#model.load_weights("my_model.h5") #loading model



model.compile(optimizer='adam',
#              optimizer=SGD(lr=0.01, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

epoch_num = 10
#epoch_num = 3 +epoch_num #Continue Training Model

with tf.device('/gpu:1'):
     history = model.fit([train_data_a, train_data_b], train_y,
              batch_size=1,
              epochs=epoch_num,
              verbose=1,
              shuffle = True,
              validation_data = ([test_data_a, test_data_b], test_y)
#              validation_split = 0.333              
              )

#MAKING HISTORY--------------------------------------- START

# list all data in history
print(history.history.keys())
# summarize history for accuracy
print('Accuracy History Plot')     
Accuracy_history = np.array([history.history['acc']])
Val_Accuracy_history = np.array([history.history['val_acc']])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plotting accuracy '+ str(epoch_num) + '.png')
plt.show()


# summarize history for loss
print('Loss History Plot') 
Loss_history = np.array([history.history['loss']])    
Val_Loss_history = np.array([history.history['val_loss']])    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plotting loss '+ str(epoch_num) + '.png')
plt.show()
#MAKING HISTORY---------------------------------------END

np.save('Accuracy_history '+ str(epoch_num), Accuracy_history)
np.save('Val_Accuracy_history '+ str(epoch_num), Val_Accuracy_history)
np.save('Loss_history '+ str(epoch_num), Loss_history)
np.save('Val_Loss_history '+ str(epoch_num), Val_Loss_history)


#MAKING HISTORY--------------------------------------- START

# list all data in history
print(history.history.keys())
# summarize history for accuracy
print('Accuracy History Plot')     
Accuracy_history = np.array([history.history['acc']])
Val_Accuracy_history = np.array([history.history['val_acc']])
plt.plot(sp.savgol_filter(history.history['acc'],31,2))
plt.plot(sp.savgol_filter(history.history['val_acc'],31,2))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('smooth plotting accuracy '+ str(epoch_num) + '.png')
plt.show()


# summarize history for loss
print('Loss History Plot') 
Loss_history = np.array([history.history['loss']])    
Val_Loss_history = np.array([history.history['val_loss']])    
plt.plot(sp.savgol_filter(history.history['loss'],31,2))
plt.plot(sp.savgol_filter(history.history['val_loss'],31,2))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('smooth plotting loss '+ str(epoch_num) + '.png')
plt.show()
#MAKING HISTORY---------------------------------------END





print('running Predicitons')
predictions = model.predict([sort_data_a, sort_data_b])
print('Predictions Finnish Running')


plt.plot(predictions)
plt.savefig('plotting epoch '+ str(epoch_num) + '.png')
plt.show()



plt.plot(predictions)
plt.plot(sort_y_binary*np.max(predictions))
plt.savefig('plotting epoch '+ str(epoch_num) + 'with true value.png')
plt.show()


print('Saving Model')
model.save('my_model.h5')
model.save('my_model_'+ str(epoch_num) + '.h5')
print('Model Saved')
'plotting epoch '+ str(epoch_num) + 'with true value.png'


test_predictions = model.predict([test_data_a, test_data_b])

plt.plot(test_predictions)
plt.plot(test_y*np.max(test_predictions))
plt.savefig('plotting epoch '+ str(epoch_num) + 'with true value.png')
plt.show()


Pred_data = np.zeros(test_predictions.shape[0])

for n in range (0,test_predictions.shape[0]):
     if test_predictions[n,0] > test_predictions[n,1]:
          Pred_data[n] = 0
     elif test_predictions[n,0] < test_predictions[n,1]:
          Pred_data[n] = 1


#flag = np.zeros(predictions.shape[0])
flag = ['']*test_predictions.shape[0]
for n in range (0,test_predictions.shape[0]):
     if test_y[n,1] == 0:
          if int(Pred_data[n]) == 0:
               flag[n] = 'TN'
          if int(Pred_data[n]) == 1:
               flag[n] = 'FP'
     elif test_y[n,1] == 1:
          if int(Pred_data[n]) == 0:
               flag[n] = 'FN'
          if int(Pred_data[n]) == 1:
               flag[n] = 'TP'
flag = np.array(flag)
df = pd.DataFrame(data ={'True_val' : test_y[:,1],
                    'Conf_Nevus'  : test_predictions[:,0],
                    'Conf_Melanoma': test_predictions[:,1],
                    'Predictions': Pred_data,
                    'Flag': flag
                    })

df.to_csv('DNN_data'+ str(epoch_num) + '.csv')
df.to_pickle('DNN_data'+ str(epoch_num) + '.pk')    

TN = sum(flag=='TN')
FN = sum(flag=='FN')
TP = sum(flag=='TP')
FP = sum(flag=='FP')

Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Diag_acc = (TP+TN)/(TP+TN+FP+FN)

print('Sensitivty : ' + str(Sensitivity*100))
print('Specificity : ' + str(Specificity*100))
print('Diagnostic Accuracy : ' + str(Diag_acc*100))
