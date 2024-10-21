import pdb, json, sys
import matplotlib.pyplot as plt
import numpy as np
import mpld3

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
        if inthere in line:
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

line1, = ax.plot(epochs, test_mean_score, label='Test Mean Score')
line2, = ax.plot(epochs, test_mean_loss, label='Test Mean Loss')

print('/'.join(filename.split('/')[:-1])+'/trainloss_gs.png')
line3, = ax.plot(epoch, train_losses, label = "train_loss")   # Plot the chart
line4, = ax.plot(epoch, train_guidance_grad_scaled, label = "train_guidance_grad_scaled")   # Plot the chart
line5, = ax.plot(epoch, train_mse_losses, label = "train_mse_losses")   # Plot the chart
line6, = ax.plot(epoch, test_mean_scores[1:], label = "val_score")   # Plot the chart
line7, = ax.plot(epoch, train_mean_scores[1:], label = "train_score")   # Plot the chart
plt.legend()

check_ax = plt.axes([0.05, 0.4, 0.1, 0.15])
check = CheckButtons(check_ax, ['Test Mean Score', 'Test Mean Loss'], [True, True])

# Function to toggle visibility
def toggle_visibility(label):
    if label == 'Test Mean Score':
        line1.set_visible(not line1.get_visible())
    elif label == 'Test Mean Loss':
        line2.set_visible(not line2.get_visible())
    if label == 'train_loss':
        line3.set_visible(not line3.get_visible())
    elif label == 'train_guidance_grad_scaled':
        line4.set_visible(not line4.get_visible())
    if label == 'train_mse_losses':
        line5.set_visible(not line5.get_visible())
    elif label == 'val_score':
        line6.set_visible(not line6.get_visible())
    if label == 'train_score':
        line7.set_visible(not line7.get_visible())
    plt.draw()


check.on_clicked(toggle_visibility)

mpld3.save_html(fig, '/'.join(filename.split('/')[:-1])+'/trainloss_epoch.png')

plt.show()

# plt.savefig('/'.join(filename.split('/')[:-1])+'/trainloss_epoch.png')  # display