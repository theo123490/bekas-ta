import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_pickle('Feature Data Coba.pk')
#data = data.sort_values(['Diagnosis'],axis=0,ascending=True)
data = data.sample(frac=1)
train_labels = data.loc[:,'Diagnosis'].values
train_labels = train_labels.astype(int)
data_a = data.drop(columns = ['Diagnosis','GLCM_mat','Normalize_Hist'])
data_a = data_a.values.astype(float)


data_b = data.loc[:,'Normalize_Hist'].values
data_b = np.concatenate(data_b)
data_b = np.reshape(data_b,(-1,4,4,4))
data_b = data_b/np.max(data_b)


data_a[:,0] = data_a[:,0]/np.max(data_a[:,0])
data_a[:,1] = data_a[:,1]/np.max(data_a[:,1])
data_a[:,2] = data_a[:,2]/np.max(data_a[:,2])
data_a[:,3] = data_a[:,3]/np.max(data_a[:,3])
data_a[:,4] = data_a[:,4]/np.max(data_a[:,4])
data_a[:,5] = data_a[:,5]/np.max(data_a[:,5])
data_a[:,6] = data_a[:,6]/np.max(data_a[:,6])
data_a[:,7] = data_a[:,7]/np.max(data_a[:,7])

data_a = np.nan_to_num(data_a)
data_b = np.nan_to_num(data_b)

y_binary = keras.utils.to_categorical(train_labels).astype(int)



train_data_a = np.concatenate([data_a[100:-100,:]])
train_data_b = np.concatenate([data_b[100:-100,:,:,:]])
train_y = np.concatenate([y_binary[100:-100,:]])

test_data_a = np.concatenate([data_a[0:100,:],data_a[-100:,:]])
test_data_b = np.concatenate([data_b[0:100,:,:,:],data_b[-100:,:,:,:]])
test_y = np.concatenate([y_binary[0:100,:],y_binary[-100:,:]])





class_names = ['Nevus', 'Melanoma']

color_input = keras.layers.Input((4,4,4))
vector_input = keras.layers.Input((8,)) 

v_dense_layer = keras.layers.Dense(10,activation="softsign")(vector_input)


color_flat_layer = keras.layers.Flatten()(color_input)


concat_layer= keras.layers.Concatenate()([v_dense_layer, color_flat_layer])
con_out_layer = keras.layers.Dense(100,activation="softsign")(concat_layer)
con_out_layer1 = keras.layers.Dense(80,activation="softsign")(con_out_layer)
con_out_layer2 = keras.layers.Dense(40,activation="softsign")(con_out_layer1)
con_out_layer3 = keras.layers.Dense(20,activation="softsign")(con_out_layer2)
output = keras.layers.Dense(2,activation="softplus")(con_out_layer3)

model = keras.models.Model(inputs=[vector_input, color_input], outputs=output)




model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])

epoch_num = 120
with tf.device('/gpu:1'):
     history = model.fit([data_a, data_b], y_binary,
              batch_size=1,
              epochs=epoch_num,
              verbose=1,
              shuffle = True,
              validation_split = 0.333
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

sort_data = data.sort_values(['Diagnosis'],axis=0,ascending=True)

sort_data_a = sort_data.drop(columns = ['Diagnosis','GLCM_mat','Normalize_Hist'])
sort_data_a = sort_data_a.values.astype(float)


sort_data_b = sort_data.loc[:,'Normalize_Hist'].values
sort_data_b = np.concatenate(sort_data_b)
sort_data_b = np.reshape(sort_data_b,(-1,4,4,4))
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
print('Model Saved')

test_loss, test_acc = model.evaluate([test_data_a, test_data_b], test_y)
print('test accuraccy : ' + str(test_acc))
print('test loss : ' + str(test_loss))

test_predictions = model.predict([test_data_a, test_data_b])


plt.plot(test_predictions)
plt.show()



plt.plot(test_predictions)
plt.plot(test_y*np.max(test_predictions))
plt.show()


