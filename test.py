import torch
from dataloader import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

net = torch.load('/home/lab70636/Projects/Tumor_DNN/model/best.pth')
testdir = '/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/val.txt'
testdata = Dataset(txtdir=testdir)
testload = DataLoader(testdata, batch_size=1)
vbar = tqdm(testload)
TP = 0
TN = 0
FP = 0
FN = 0

for iter, (indata, label) in enumerate(vbar):
    pred = net(indata)
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

sensitivity = TP / TP + FN
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print('sensitivity:{:.4f} specificity:{:.4f} accuracy:{:.4f}\n'.format(
    sensitivity,
    specificity,
    accuracy))
