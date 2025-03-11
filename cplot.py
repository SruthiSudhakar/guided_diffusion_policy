import pdb, json, sys
import matplotlib.pyplot as plt
import numpy as np

# Using readlines()
filename = sys.argv[1]
file1 = open(filename, 'r')
Lines = file1.readlines()

train_losses=[]
val_losses=[0]
valid_accuracy=[0]
global_step=[]
epoch=[]
# Strips the newline character
for line in Lines:
    line=json.loads(line)
    train_losses.append(line['train_loss'])
    global_step.append(line['global_step'])
    epoch.append(line['epoch'])
    if 'valid_accuracy' in line:
        val_losses.append(line['val_loss'])
        valid_accuracy.append(line['valid_accuracy'])
    else:
        val_losses.append(val_losses[-1])
        valid_accuracy.append(valid_accuracy[-1])

print('/'.join(filename.split('/')[:-1])+'/trainloss_gs.png')
plt.plot(epoch, train_losses, label = "train_loss")   # Plot the chart
plt.plot(epoch, val_losses[1:], label = "val_loss")   # Plot the chart
plt.plot(epoch, valid_accuracy[1:], label = "valid_accuracy")   # Plot the chart
plt.legend()
plt.savefig('/'.join(filename.split('/')[:-1])+'/trainloss_epoch.png')  # display