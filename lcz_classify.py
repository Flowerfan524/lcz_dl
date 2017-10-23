import torch
import data.LCZDataset as LCZD
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50


def get_data(img_file, label_file, height, width, batch_size):
    transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = LCZD(img_file, label_file, transform=transformer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for data in data_loader:
        images, labels = data['image'], data['label']
        outputs = model(Variable(images))
        _, Predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct == (Predicted == labels).sum()
    print('Accuracy of the network is %.3f' % (correct / total))


def main():
    img_file = ''
    label_file = ''
    height, width = 32, 32
    batch_size = 32
    num_epochs = 50
    data_loader = get_data(img_file, label_file, height, width, batch_size)
    model = resnet50(pretrained=True, num_classes=17)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):
        running_loss = 0
        for i, sample in enumerate(data_loader):
            inputs, labels = sample['image'], sample['label']
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 20 == 19:
                print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0
    evaluate(model, data_loader=data_loader)





if __name__ == '__main__':
    main()
