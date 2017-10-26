import torch
from data import LCZDataset as LCZD
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from resnet import resnet18, resnet50
import numpy as np
import pdb
import collections

def get_data(img_file, label_file):
    imgs = np.load(img_file)['arr_0'].astype('float32')
    labels = np.load(label_file)['arr_0'].astype('int') - 1
    return imgs, labels

def get_dataloader(imgs,labels,batch_size=64):
    transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_set = LCZD(imgs, labels, transform=transformer)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader


def evaluate(model, data_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    for data in data_loader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        _, Predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (Predicted == labels.cuda()).sum()
    print('Accuracy of the network is %.3f' % (correct / total))

def split_data(imgs,labels,ratio=0.7):
    assert len(imgs) == len(labels)
    ids = np.arange(len(labels))
    np.random.shuffle(ids)
    num_train = int(len(labels) * ratio)
    train_imgs,train_labels = imgs[:num_train],labels[:num_train]
    test_imgs, test_labels = imgs[num_train:],labels[num_train:]
    return train_imgs,train_labels,test_imgs,test_labels

def main():
    img_file = 'hk_imgs.npz'
    label_file = 'hk_labels.npz'
    wh_file = 'wh_imgs.npz'
    wh_label = 'wh_label.npz'
    xa_file = 'xa_img.npz'
    xa_label = 'xa_label.npz'
    height, width = 32, 32
    batch_size = 256
    num_epochs = 30
    hk_imgs,hk_labels = get_data(img_file,label_file)
    wh_imgs,wh_labels = get_data(wh_file,wh_label)
    xa_imgs,xa_labels = get_data(xa_file,xa_label)
    data1 = split_data(hk_imgs,hk_labels)
    data2 = split_data(wh_imgs,wh_labels)
    data3 = split_data(xa_imgs,xa_labels)
    print('hk',collections.Counter(hk_labels))
    print('wh',collections.Counter(wh_labels))
    print(collections.Counter(xa_labels))
    #pdb.set_trace() 
    train_imgs = np.concatenate((data1[0],data2[0],data3[0]))
    train_labels = np.concatenate((data1[1],data2[1],data3[1]))
    test_imgs = np.concatenate((data1[2],data2[2]))
    test_labels = np.concatenate((data1[3],data2[3]))
    train_loader = get_dataloader(train_imgs,train_labels)
    #print(len(data2[0]),len(data2[1]))
    train_loader = get_dataloader(train_imgs,train_labels)
    test_loader = get_dataloader(test_imgs,test_labels)
    test_loader1 = get_dataloader(data1[2],data1[3])
    test_loader2 = get_dataloader(data2[2],data2[3])
    test_loader3 = get_dataloader(data3[2],data3[3])
    
    #test_loader = get_data(wh_file, wh_label, height, width, batch_size)
    model_ft =  resnet50(pretrained=True,num_classes=17)
    model_ft.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):
        running_loss = 0
        for i, sample in enumerate(train_loader):
            inputs, labels = sample
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 20 == 19:
                print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0
                evaluate(model_ft, data_loader=test_loader)
                evaluate(model_ft, data_loader=test_loader1)
                evaluate(model_ft, data_loader=test_loader2)
                evaluate(model_ft, data_loader=test_loader3)
    #evaluate(model_ft, data_loader=data_loader)
    evaluate(model_ft, data_loader=test_loader)
    evaluate(model_ft, data_loader=test_loader1)
    evaluate(model_ft, data_loader=test_loader2)
    evaluate(model_ft, data_loader=test_loader3)





if __name__ == '__main__':
    main()
