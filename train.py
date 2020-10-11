import time
import torch
import torch.nn as nn
from tqdm import tqdm
from tumor_DNN import USDNN
from dataloader import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

epochs = 100
batch_size = 1326
lr = 0.0001
train_data = Dataset(imgdir='/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/train_seg_deeplab/',
                     exceldir='/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/2018LN.xlsx')
val_data = Dataset(imgdir='/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/val_seg_deeplab/',
                   exceldir='/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/2018LN.xlsx')
save_dir = '/home/lab70636/Projects/Tumor_DNN/model/'
writer = SummaryWriter('/home/lab70636/Projects/Tumor_DNN/log')

net = USDNN().cuda()
net1 = USDNN().cuda()
net2 = USDNN().cuda()
net3 = USDNN().cuda()
net4 = USDNN().cuda()
netlist = [net, net1, net2, net3, net4]
loss_func = nn.CrossEntropyLoss()

train_load = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_load = DataLoader(val_data, batch_size=1, shuffle=False)
maxacc = 0
for epoch in range(epochs):
    tbar = tqdm(train_load)
    for model in netlist:
        model.cuda()
    for iter, (indata, label) in enumerate(tbar):
        indata = indata.cuda()
        label = label.cuda()
        tlosslist = []
        for model in netlist:
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            pred = model(indata)
            loss = loss_func(pred, label).cuda()
            tlosslist.append(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

    tbar.close()
    tloss = sum(tlosslist) / len(tlosslist)
    tloss = loss.data.cpu().numpy()
    writer.add_scalar('train_loss', tloss, epoch)

    vbar = tqdm(val_load)
    for model in netlist:
        model.eval()
    val_loss = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for iter, (indata, label) in enumerate(vbar):
        indata = indata.cuda()
        label = label.cuda()
        multi_pred = []
        multi_vloss = []
        for model in netlist:
            pred = model(indata)
            multi_pred.append(pred)
            vloss = loss_func(pred, label).cuda()
            multi_vloss.append(vloss)
        vloss = sum(multi_vloss) / len(multi_vloss)
        val_loss.append(vloss.data.cpu().numpy())
        pred = sum(multi_pred) / len(multi_pred)
        pred = pred.data.cpu().numpy()[0]
        label = label.data.cpu().numpy()[0]
        if label == 1 and pred[label] > pred[abs(label - 1)]:
            TP += 1
        elif label == 1 and pred[label] < pred[abs(label - 1)]:
            FN += 1
        elif label == 0 and pred[label] > pred[abs(label - 1)]:
            TN += 1
        elif label == 0 and pred[label] < pred[abs(label - 1)]:
            FP += 1
    vbar.close()

    avg_vloss = sum(val_loss) / len(val_loss)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    writer.add_scalar('val_loss', avg_vloss, epoch)
    writer.add_scalar('sensitivity', sensitivity, epoch)
    writer.add_scalar('specificity', specificity, epoch)
    writer.add_scalar('accuracy', accuracy, epoch)
    writer.flush()
    # time.sleep(0.5)
    print('\nepoch:{}/{} tloss:{:.4f} vloss:{:.4f} sensitivity:{:.4f} specificity:{:.4f} accuracy:{:.4f}\n'.format(
        epoch,
        epochs - 1,
        tloss,
        avg_vloss,
        sensitivity,
        specificity,
        accuracy))

    if accuracy >= maxacc:
        maxacc = accuracy
        bestname = save_dir + 'best'
        f = open(save_dir + 'best.txt', 'w')
        f.writelines(
            'epoch:{}/{} tloss:{:.4f} vloss:{:.4f} sensitivity:{:.4f} specificity:{:.4f} accuracy:{:.4f}\n'.format(
                epoch,
                epochs - 1,
                tloss,
                avg_vloss,
                sensitivity,
                specificity,
                accuracy))
        f.close()
        torch.save(net, bestname + '1.pth')
        torch.save(net1, bestname + '2.pth')
        torch.save(net2, bestname + '3.pth')
        torch.save(net3, bestname + '4.pth')
        torch.save(net4, bestname + '5.pth')
    if epoch != 0 and (epoch + 1) % 10 == 0:
        filename = save_dir + str(epoch) + '.pth'
        torch.save(net, filename)
writer.close()
