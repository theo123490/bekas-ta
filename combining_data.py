import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

num_1 = 5000
num_2 = 800
new_num = num_1+num_2

loss_hist_1 = np.load('Loss_history '+ str(num_1)+'.npy')
loss_hist_2 = np.load('Loss_history '+ str(new_num)+'.npy')
loss_new_hist = np.concatenate((loss_hist_1, loss_hist_2), axis=None)

acc_hist_1 = np.load('Accuracy_history '+ str(num_1)+'.npy')
acc_hist_2 = np.load('Accuracy_history '+ str(new_num)+'.npy')
acc_new_hist = np.concatenate((acc_hist_1, acc_hist_2), axis=None)

val_loss_hist_1 = np.load('Val_Loss_history '+ str(num_1)+'.npy')
val_loss_hist_2 = np.load('Val_Loss_history '+ str(new_num)+'.npy')
val_loss_new_hist = np.concatenate((val_loss_hist_1, val_loss_hist_2), axis=None)

val_acc_hist_1 = np.load('Val_Accuracy_history '+ str(num_1)+'.npy')
val_acc_hist_2 = np.load('Val_Accuracy_history '+ str(new_num)+'.npy')
val_acc_new_hist = np.concatenate((val_acc_hist_1, val_acc_hist_2), axis=None)

#loss_new_hist = loss_new_hist[0,:]
#val_loss_new_hist = val_loss_new_hist[0,:]
#acc_new_hist = acc_new_hist[0,:]
#val_acc_new_hist = val_acc_new_hist[0,:]


np.save('Accuracy_history '+ str(new_num), acc_new_hist)
np.save('Val_Accuracy_history '+ str(new_num), val_acc_new_hist)
np.save('Loss_history '+ str(new_num), loss_new_hist)
np.save('Val_Loss_history '+ str(new_num), val_loss_new_hist)



#-------------------------------------------------------------------------


plt.plot(acc_new_hist)
plt.plot(val_acc_new_hist)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plotting accuracy '+ str(new_num) + '.png')
plt.show()


plt.plot(loss_new_hist)
plt.plot(val_loss_new_hist)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plotting loss '+ str(new_num) + '.png')
plt.show()


plt.plot(sp.savgol_filter(acc_new_hist,31,2))
plt.plot(sp.savgol_filter(val_acc_new_hist,31,2))

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('smooth plotting accuracy '+ str(new_num) + '.png')
plt.show()


plt.plot(sp.savgol_filter(loss_new_hist,31,2))
plt.plot(sp.savgol_filter(val_loss_new_hist,31,2))

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('smooth plotting loss '+ str(new_num) + '.png')
plt.show()


