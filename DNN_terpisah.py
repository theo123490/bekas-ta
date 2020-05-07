import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp




sort_data_a = np.load('sort_data_a.npy')
sort_data_b = np.load('sort_data_b.npy')
sort_y_binary = np.load('sort_y_binary.npy')

train_data_a = np.load('train_data_a.npy')
train_data_b = np.load('train_data_b.npy')
train_y = np.load('train_y.npy')

test_data_a = np.load('test_data_a.npy')
test_data_b = np.load('test_data_b.npy')
test_y = np.load('test_y.npy')



class_names = ['Nevus', 'Melanoma']


kernel_initializer_input = 'glorot_normal'
bias_initializer_input='zeros'


color_input = keras.layers.Input((16,16,16))
vector_input = keras.layers.Input((98,)) 
#v_dense_layer = keras.layers.Dense(100,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(vector_input)
#v_dense_layer1 = keras.layers.Dense(100,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(v_dense_layer)



color_flat_layer = keras.layers.Flatten()(color_input)
#color_dense = keras.layers.Dense(100,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(color_flat_layer)
#color_dense1 = keras.layers.Dense(100,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(color_dense)



#concat_layer= keras.layers.Concatenate()([v_dense_layer1, color_dense1])
concat_layer= keras.layers.Concatenate()([vector_input, color_flat_layer])
con_out_layer1 = keras.layers.Dense(900,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(concat_layer)
con_out_layer2 = keras.layers.Dense(900,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer1)
con_out_layer3 = keras.layers.Dense(800,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer2)
con_out_layer4 = keras.layers.Dense(700,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer3)
con_out_layer5 = keras.layers.Dense(600,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer4)
con_out_layer6 = keras.layers.Dense(600,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer5)
con_out_layer7 = keras.layers.Dense(400,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer6)
con_out_layer8 = keras.layers.Dense(200,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer7)
con_out_layer9 = keras.layers.Dense(100,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer8)
con_out_layer10 = keras.layers.Dense(70,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer9)
con_out_layer11 = keras.layers.Dense(50,activation="tanh",kernel_initializer=kernel_initializer_input,bias_initializer=bias_initializer_input)(con_out_layer10)

output = keras.layers.Dense(2,activation="sigmoid",kernel_initializer=kernel_initializer_input)(con_out_layer10)

model = keras.models.Model(inputs=[vector_input, color_input], outputs=output)
#model.load_weights("best_model_acc_'+str(epoch_num)+'.h5") #loading model



model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False),
#              optimizer=SGD(lr=0.01, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])


epoch_num = 300
#epoch_num = 3 +epoch_num #Continue Training Model


save_best_val_acc = keras.callbacks.ModelCheckpoint('best_model_val_acc_'+str(epoch_num)+'.h5', 
                                           save_best_only=True, 
                                           monitor='val_acc', 
                                           mode='max')

save_best_acc = keras.callbacks.ModelCheckpoint('best_model_acc_'+str(epoch_num)+'.h5', 
                                           save_best_only=True, 
                                           monitor='acc', 
                                           mode='max')


with tf.device('/gpu:1'):
     history = model.fit([train_data_a, train_data_b], train_y,
              batch_size= 150,
              epochs=epoch_num,
              verbose=1,
              shuffle = True,
              validation_data = ([test_data_a, test_data_b], test_y),
              callbacks = [save_best_acc, save_best_val_acc]
              )

#MAKING HISTORY---------------------------------------------------------------------------- START
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
#MAKING HISTORY---------------------------------------------------------------------------- END



def run_eval():
     #EVALUATION--------------------------------------------------------------------------------START
     #PREDICTIONS---------------------------------------------START
     
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
     #PREDICTIONS---------------------------------------------END
     
     
     print('Saving Model')
     model.save('my_model.h5')
     model.save('my_model_'+ str(epoch_num) + '.h5')
     print('Model Saved')
     'plotting epoch '+ str(epoch_num) + 'with true value.png'
     
     
     
     #EVALUATION PARAMETERS-----------------------------------------------START
     #TEST DATA PROCESSING---------------------------------------START
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
     
     print('TEST_Sensitivty : ' + str(Sensitivity*100))
     print('TEST_Specificity : ' + str(Specificity*100))
     print('TEST_Diagnostic Accuracy : ' + str(Diag_acc*100))
     #TEST DATA PROCESSING---------------------------------------END
     
     
     
     
     
     #TRAIN DATA PROCESSING---------------------------------------START
     train_predictions = model.predict([train_data_a, train_data_b])
     
     plt.plot(train_predictions)
     plt.plot(train_y*np.max(train_predictions))
     plt.show()
     
     
     train_Pred_data = np.zeros(train_predictions.shape[0])
     
     for n in range (0,train_predictions.shape[0]):
          if train_predictions[n,0] > train_predictions[n,1]:
               train_Pred_data[n] = 0
          elif train_predictions[n,0] < train_predictions[n,1]:
               train_Pred_data[n] = 1
     
     
     #flag = np.zeros(predictions.shape[0])
     train_flag = ['']*train_predictions.shape[0]
     for n in range (0,train_predictions.shape[0]):
          if train_y[n,1] == 0:
               if int(train_Pred_data[n]) == 0:
                    train_flag[n] = 'TN'
               if int(train_Pred_data[n]) == 1:
                    train_flag[n] = 'FP'
          elif train_y[n,1] == 1:
               if int(train_Pred_data[n]) == 0:
                    train_flag[n] = 'FN'
               if int(train_Pred_data[n]) == 1:
                    train_flag[n] = 'TP'
     train_flag = np.array(train_flag)
#     train_df = pd.DataFrame(data ={'True_val' : train_y[:,1],
#                         'Conf_Nevus'  : train_predictions[:,0],
#                         'Conf_Melanoma': train_predictions[:,1],
#                         'Predictions': train_Pred_data,
#                         'Flag': train_flag
#                         })
     
     #df.to_csv('train_DNN_data.csv')
     #df.to_pickle('train_DNN_data.pk')    
     
     train_TN = sum(train_flag=='TN')
     train_FN = sum(train_flag=='FN')
     train_TP = sum(train_flag=='TP')
     train_FP = sum(train_flag=='FP')
     
     train_Sensitivity = train_TP/(train_TP+train_FN)
     train_Specificity = train_TN/(train_TN+train_FP)
     train_Diag_acc = (train_TP+train_TN)/(train_TP+train_TN+train_FP+train_FN)
     
     print('train_Sensitivty : ' + str(train_Sensitivity*100))
     print('train_Specificity : ' + str(train_Specificity*100))
     print('train_Diagnostic Accuracy : ' + str(train_Diag_acc*100))
     #TRAIN DATA PROCESSING---------------------------------------END
     #EVALUATION PARAMETERS--------------------------------------------------------------END
     #EVALUATION--------------------------------------------------------------------------------END

run_eval()