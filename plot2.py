import pdb, json, sys
import matplotlib.pyplot as plt
import numpy as np
import mpld3
from matplotlib.widgets import CheckButtons
# Using readlines()
filename = sys.argv[1]
file1 = open(filename, 'r')
Lines = file1.readlines()

train_losses=[0]
global_step=[0]
train_guidance_grad_scaled=[0]
train_mse_losses=[0]
test_mean_scores=[0]
train_mean_scores=[0]
epoch=[0]
# Strips the newline character
inthere = 'train_guidance_grad_scaled' in Lines[0]
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
        if inthere:
            train_guidance_grad_scaled.append(line['train_guidance_grad_scaled'])
            train_mse_losses.append(line['train_mse_losses'])
        else:
            train_guidance_grad_scaled.append(0)
            train_mse_losses.append(0)
        global_step.append(line['global_step'])
        epoch.append(line['epoch'])
    except:
        train_losses.append(train_losses[-1])
        train_guidance_grad_scaled.append(line[-1])
        train_mse_losses.append(line[-1])
        global_step.append(global_step[-1])
        epoch.append(epoch[-1])
train_losses=train_losses[1:]
train_guidance_grad_scaled=train_guidance_grad_scaled[1:]
train_mse_losses=train_mse_losses[1:]

global_step=global_step[1:]
epoch=epoch[1:]

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2)

line4, = ax.plot(epoch, train_guidance_grad_scaled, label = "train_guidance_scaled")   # Plot the chart
line5, = ax.plot(epoch, train_mse_losses, label = "train_mse_losses")   # Plot the chart
plt.legend()

plt.show()

plt.savefig('/'.join(filename.split('/')[:-1])+'/train_terms_epoch.png')  # display
