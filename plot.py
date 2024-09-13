import pdb, json, sys
import matplotlib.pyplot as plt
import numpy as np

# Using readlines()
filename = sys.argv[1]
file1 = open(filename, 'r')
Lines = file1.readlines()

train_losses=[]
global_step=[]
test_mean_scores=[0]
train_mean_scores=[0]
epoch=[]
# Strips the newline character
for line in Lines:
    if 'test/mean_score' in line:
        try:
            test_mean_scores.append(json.loads(line)['test/mean_score'])
            train_mean_scores.append(json.loads(line)['train/mean_score'])
        except:
            test_mean_scores.append(float(line.split('test/mean_score": ')[-1][:3]))
            train_mean_scores.append(float(line.split('train/mean_score": ')[-1][:3]))
    else:
        test_mean_scores.append(test_mean_scores[-1])
        train_mean_scores.append(train_mean_scores[-1])
    try:
        line=json.loads(line)
        # print(line)
        train_losses.append(line['train_loss'])
        global_step.append(line['global_step'])
        epoch.append(line['epoch'])
    except:
        train_losses.append(train_losses[-1])
        global_step.append(global_step[-1])
        epoch.append(epoch[-1])
plt.plot(global_step, train_losses, label = "train_loss")   # Plot the chart
plt.plot(global_step, test_mean_scores[1:], label = "val_score")   # Plot the chart
plt.plot(global_step, train_mean_scores[1:], label = "train_score")   # Plot the chart
plt.legend()
plt.savefig('/'.join(filename.split('/')[:-1])+'/trainloss_gs.png')  # displayplt.plot(global_step, train_losses, label = "train_loss")   # Plot the chart
plt.clf()
print('/'.join(filename.split('/')[:-1])+'/trainloss_gs.png')
plt.plot(epoch, train_losses, label = "train_loss")   # Plot the chart
plt.plot(epoch, test_mean_scores[1:], label = "val_score")   # Plot the chart
plt.plot(epoch, train_mean_scores[1:], label = "train_score")   # Plot the chart
plt.legend()
plt.savefig('/'.join(filename.split('/')[:-1])+'/trainloss_epoch.png')  # display