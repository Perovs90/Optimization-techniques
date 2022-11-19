
# # Optimization Techniques - Classification of X-Ray Images 


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, random_split
import torch.nn.functional as F

import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import skimage
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score


# ## Data preparation

#defining data augmentation
def data_transforms(phase):
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    if phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    if phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])             
    return transform

#enable GPU to compute mini batches
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#setting the directory of the data folder
#Data is described here: Daniel Kermany, Kang Zhang, Michael Goldbaum. (2018). Labeled Optical
#Coherence Tomography (OCT) and Chest X-Ray Images for Classification
path = 'chest_xray'

#showing bar charts with class distribution
train_samplesize = pd.DataFrame.from_dict(
    {'Normal': [len([os.path.join(path+'/train/NORMAL', filename) 
                     for filename in os.listdir(path+'/train/NORMAL')])], 
     'Pneumonia': [len([os.path.join(path+'/train/PNEUMONIA', filename) 
                        for filename in os.listdir(path+'/train/PNEUMONIA')])]})
sns.barplot(data=train_samplesize).set_title('Training Set Data Inbalance', fontsize=20)
plt.show()


n_norm = train_samplesize['Normal'].iloc[0]
n_pneum = train_samplesize['Pneumonia'].iloc[0]
print(f'Amount of "Normal": {n_norm}')
print(f'Amount of "Pneumonia": {n_pneum}')


#upload images from the folders
image_datasets = {x: datasets.ImageFolder(os.path.join(path, x), data_transforms(x)) 
                  for x in ['train', 'val', 'test']}


#Create dataloaders for iterative access to images (full data set)
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 5, shuffle=True), 
               'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size = 1, shuffle=True), 
               'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 1, shuffle=True)}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

classes = image_datasets['train'].classes


#### To train on short trainset (1400 examples) execute following cells:

train_new, _ =  random_split(image_datasets['train'], [1400, 3474], generator=torch.Generator().manual_seed(42))

#To sample balanced dataset
class_weights = [3, 1]
sample_weights = [0]*len(train_new)
for idx, (data, label) in enumerate(train_new):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
        
sampler = WeightedRandomSampler(sample_weights, num_samples=
                                    len(sample_weights), replacement=True)

dataloaders = {'train': torch.utils.data.DataLoader(train_new, batch_size = len(train_new), sampler = sampler), 
               'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size = 1, shuffle=True), 
               'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 1, shuffle=True)}
######


# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize = (20,2))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(dataloaders['train'])#returns an iterator for the given function
images, labels = dataiter.next()

batch_size = 5
# show images
imshow(torchvision.utils.make_grid(images[:]))
# print labels
print(' '.join(f'{classes[labels[j]]} ({labels[j]})' for j in range(batch_size)))
####

# ### Pretrained model VGG

#To get weights of the pretrained model VGG16:
#model = models.vgg16(pretrained=True)
#torch.save(model.state_dict(), 'model_weights.pth')
#To upload weights of the pretrained model VGG16 (previously saved):
model_pre = models.vgg16() # we do not specify pretrained=True
model_pre.load_state_dict(torch.load('model_weights.pth'))

#Adding the fully connected layer to train on our data:

for param in model_pre.features.parameters():
    param.required_grad = False

num_features = model_pre.classifier[6].in_features
features = list(model_pre.classifier.children())[:-1] 
features.extend([nn.Linear(num_features, len(class_names))])
model_pre.classifier = nn.Sequential(*features) 
print(model_pre)


# ### Custom simple Neural Network

#Define Custom Convolutional Neural Network

class Net(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 5) #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        self.conv3 = nn.Conv2d(16, 10, 5)
        self.fc1 = nn.Linear(360, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#If you plan to use custom neural network:
net = Net()
net.to(device)

#If you plan to train pretrained VGG model:
net = model_pre
net.to(device)

num_epochs = 12

#Define function for training of NN with SGD and ADAM
def network_training(net, optimizer, criterion, num_epochs = num_epochs):
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    start_time = time.time()
    loss_list = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            running_corrects = 0 
            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # print statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels)

                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    loss_list += [running_loss / 200]
                    running_loss = 0.0
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{}  Acc: {:.4f}'.format(phase, epoch_acc))
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
    training_time = time.time() - start_time
    print(f'time of training: {training_time}')
    print('Best val Acc: {:4f}'.format(best_acc))
    net.load_state_dict(best_model_wts)
    return net, loss_list, training_time

###function with full evaluation procedure:
def evaluation_of_net():
    n_train = len(dataloaders['train'].dataset)
    n_test = len(dataloaders['test'].dataset)
    
    missclassified_train = 0
    net.train(False)
    for i, data in enumerate(dataloaders['train'], 0):
        inputs, label = data[0], data[1]
        inputs = inputs.to(device)
        label = label.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.to(device)
        missclassified_train += torch.sum(predicted != label).item()
    classification_error_train = missclassified_train/n_train*100

    missclassified_test = 0
    net.train(False)
    labels = []
    predictions = []
    for i, data in enumerate(dataloaders['test'], 0):
        inputs, label = data[0], data[1]
        labels += label
        inputs = inputs.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.to('cpu')
        predictions += predicted
        missclassified_test += torch.sum(predicted != label).item()
    classification_error_test = missclassified_test/n_test*100
    
    print('Results for optimizer:')
    print(f'classification error train: {classification_error_train}')
    print(f'classification error test: {classification_error_test}')
    print(f'accuracy: {accuracy_score(labels, predictions)}')
    print(f'auc: {roc_auc_score(labels, predictions)}')
    print(f'f1: {f1_score(labels, predictions)}')
    print(f'recall: {recall_score(labels, predictions)}')
    print(f'precision: {precision_score(labels, predictions)}')
    print(f'training time: {training_time}')
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    ax = sns.heatmap(cm, annot=True, fmt="d")
    return classification_error_train, classification_error_test, labels, predictions


# ## SGD optimizer

criterion = nn.CrossEntropyLoss(weight = torch.tensor([2.0, 1.0])).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.01)

#to get network trained:
net, loss_SGD, training_time = network_training(net, optimizer, criterion)
#to get the evaluation results:
classification_error_train, classification_error_test, labels, predictions = evaluation_of_net()

# Results for SGD optimizer (1400 examples balanced):
# - classification error train: 2.7857142857142856
# - classification error test: 13.099415204678364
# - accuracy: 0.8690058479532163
# - auc: 0.8760545905707197
# - f1: 0.8694638694638696
# - recall: 0.9564102564102565
# - precision: 0.7970085470085471
# - training time: 455.47358679771423


#to plot graphs:
epochs = [0.2*(x+1) for x in range(num_epochs*5)]


# To show images from augmented dataset

dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(5)))


# ### Results optim.SGD(net.parameters(), lr=0.001, momentum=0.9), whole trainset (5003 exanples):
# * classification error train: 3.4579252448530884
# * classification error test: 14.502923976608187
# * accuracy: 0.8549707602339182
# * auc: 0.8652191894127378
# * f1: 0.8606741573033707
# * recall: 0.982051282051282
# * precision: 0.766
# * tn - 348, tp - 383, false negative - 117, false positive - 7
# * training time: 577.2731721401215

# ### Results for SGD optimizer (1400 examples, 8 epochs):
# - classification error train: 4.071428571428572
# - classification error test: 13.099415204678364
# - accuracy: 0.8690058479532163
# - auc: 0.8760545905707197
# - f1: 0.8694638694638696
# - recall: 0.9564102564102565
# - precision: 0.7970085470085471
# - tn - 370, fp - 95, fn - 17, tp - 373
# - training time: 327.6969985961914

# ## ADAM optimizer

optimizer = optim.Adam(net.parameters(), lr=0.001)


#1400 balanced
net_ADAM, loss_ADAM, training_time_ADAM  = network_training(net, optimizer, criterion)

classification_error_train, classification_error_test, labels, predictions = evaluation_of_net()


#to plot graphs:
epochs = [0.2*(x+1) for x in range(num_epochs*5)]
loss_SGD = [0.667130921036005, 0.6177664376050234, 0.44886718412628396, 0.29078783782126266, 0.233370353821374, 0.2092554322001524, 0.2079826455481816, 0.16494992157968227, 0.1889635531057138, 0.18631241600436624, 0.17475722527131438, 0.17969541036291048, 0.15284368914697552, 0.14548640505410732, 0.1784410884929821, 0.13653506067188573, 0.13217269032407786, 0.14822106849169359, 0.1140954573644558, 0.16370714601071085]
loss_ADAM = [0.3430385389365256, 0.21929042096133344, 0.2219471736389096, 0.20555738755821948, 0.18728539881762118, 0.19128660976770334, 0.1572097970910545, 0.17124671252560802, 0.1636290114223084, 0.15694865922523604, 0.1750800750761118, 0.1317783482206869, 0.15451775662600994, 0.1362634438488749, 0.1025300239813805, 0.13601013270326803, 0.12209524801161024, 0.12207305029714917, 0.1191797923442573, 0.09953186067768911]

fig, ax = plt.subplots()
ax.plot(epochs, loss_SGD, label = 'SGD') #labels
ax.plot(epochs, loss_ADAM, label = 'ADAM') #labels
ax.legend()
ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('loss', fontsize=14)

# ### Results optim.Adam(net.parameters(), lr=0.001) (5003 examples, 4 epochs):
# * classification error train: 6.935838496901859
# * classification error test: 9.707602339181287
# * accuracy: 0.9029239766081871
# * auc: 0.9055831265508685
# * f1: 0.8979089790897908
# * recall: 0.9358974358974359
# * precision: 0.8628841607565012
# * tn - 407, tp - 365, fn - 58, fp - 25
# * training time: 577.2731721401215

# ### Results for ADAM optimizer (1400 examples, 8 epochs):
# - classification error train: 9.142857142857142
# - classification error test: 12.397660818713451
# - accuracy: 0.8760233918128655
# - auc: 0.8777502067824648
# - f1: 0.8684863523573202
# - recall: 0.8974358974358975
# - precision: 0.8413461538461539
# - tn - 399, fp - 66, fp - 40, tp - 350
# - training time: 327.6969985961914
# loss = [0.350, 0.230, 0.182, 0.160, 0.143, 0.125, 0.099, 0.064]

# ## LBFGS optimizer

# This is a very memory intensive optimizer (it requires additional param_bytes * (history_size + 1) bytes). If it doesn’t fit in memory try reducing the history size, or use a different algorithm.

optimizer = optim.LBFGS(net.parameters(), history_size=3, max_iter=4)
criterion = nn.CrossEntropyLoss(weight = torch.tensor([2.0, 1.0]).to(device))


# We need to rewrite the training function for this optimizer

def network_training_LBFGS(optimizer, criterion, num_epochs = num_epochs):
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    start_time = time.time()
    loss_list = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
               # scheduler.step()
                net.train()
            else:
                net.eval()
            running_corrects = 0
            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                def closure():
                    optimizer.zero_grad()
                    outputs = net(inputs).to(device)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss
                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs).to(device)
                    _, preds = torch.max(outputs, 1)     
                    if phase == 'train':
                        optimizer.step(closure)
                # print statistics
                loss = criterion(outputs, labels)
                running_loss += loss.item()* inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
    training_time = time.time() - start_time
    print(f'time of training: {training_time}')
    print('Best val Acc: {:4f}'.format(best_acc))
    net.load_state_dict(best_model_wts)
    return net, training_time

#To train and evaluate:
net, training_time  = network_training_LBFGS(optimizer, criterion)
classification_error_train, classification_error_test, labels, predictions = evaluation_of_net()

#Results for LBFGS
#classification error train: 8.071428571428571
#classification error test: 17.77777777777778
#accuracy: 0.8222222222222222
#auc: 0.8303556658395368
#f1: 0.8256880733944953
#recall: 0.9230769230769231
#precision: 0.7468879668049793
#tn - 333 fn -120 fp- 30 tp - 360
#training time: 1089




#Final plot
loss_SGD = [0.6562243689596653,
 0.4599374588858336,
 0.23101389307063072,
 0.19068763130053412,
 0.18702555026859044,
 0.2004616828696453,
 0.1767051037074998,
 0.14657878553902265]
loss_ADAM = [0.350, 0.230, 0.182, 0.160, 0.143, 0.125, 0.099, 0.064]
loss_LBFGS = [0.7235, 0.6497, 0.6255, 0.5378, 0.3766, 0.2742, 0.2372, 0.2260]
epochs = [x+1 for x in range(8)]

fig, ax = plt.subplots()
ax.plot(epochs, loss_SGD, label = 'SGD') #labels
ax.plot(epochs, loss_ADAM, label = 'ADAM') #labels
ax.plot(epochs, loss_LBFGS, label = 'LBFGS') #labels
ax.legend()
ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('loss', fontsize=14)

###For pretrained models:

#Results Pretrained ADAM (5003 examples, 4 epochs)
#classification error train: 3.118129122526484
#classification error test: 9.707602339181287
#accuracy: 0.9029239766081871
#auc: 0.9093052109181141
#f1: 0.9022379269729092
#recall: 0.982051282051282
#precision: 0.8344226579520697
#tn - 389, fp - 76, fn - 7, tp - 383
#training time: 3315.048468351364

#Results Pretrained SGD (5003 examples, 4 epochs)
#classification error train: 1.9588247051768939
#classification error test: 9.473684210526317
#accuracy: 0.9052631578947369
#auc: 0.9122828784119107
#f1: 0.9052631578947369
#recall: 0.9923076923076923
#precision: 0.832258064516129
#training time: 3497.309763431549
epochs = [0.2*(x+1) for x in range(num_epochs*5)]
loss_ADAM = [0.4962979547815739, 0.39942742579530127, 0.2583397812021558, 0.13232108810240548, 0.20342186968057244, 0.14410376537742312, 0.2337562293381019, 0.1382066022816975, 0.08967739214762802, 0.1142085158525597, 0.11271950261411721, 0.05761351530442973, 0.06914991394510767, 0.10124207890264794, 0.08687108471477518, 0.10293307367281417, 0.07306733721273885, 0.07947615258442715, 0.05737379003592974, 0.05930434423430235]
loss_SGD = [0.24005027471648646, 0.1525964652977518, 0.14940176398733457, 0.10844821355711247, 0.13954504554218147, 0.10211493664071895, 0.0764560849168629, 0.09138071909077553, 0.09655824798697722, 0.07286104824815993, 0.0588610545074198, 0.09113316440218114, 0.056214909627592534, 0.07071654269171632, 0.056139844164626994, 0.04143137010884857, 0.05791820376736098, 0.08738085085787134, 0.0582066200895315, 0.065875858635236]
fig, ax = plt.subplots()
ax.plot(epochs, loss_SGD, label = 'SGD') #labels
ax.plot(epochs, loss_ADAM, label = 'ADAM') #labels
ax.legend()
ax.set_xlabel('epoch', fontsize=14)
ax.set_ylabel('loss', fontsize=14)