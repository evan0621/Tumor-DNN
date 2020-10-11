import os
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from skimage.feature import greycomatrix, greycoprops


class Dataset(Dataset):
    def __init__(self, imgdir, exceldir):
        self.exceldir = exceldir
        self.imgdir = imgdir
        self.excel = pd.read_excel(self.exceldir)
        self.imglist = os.listdir(imgdir)
        self.grey_level = 128
        self.distance = 1
        self.datalist = np.zeros((1326, 34), dtype='float32')
        self.labelist = np.zeros(1326, dtype='int')

    def __getitem__(self, index):
        if self.datalist[index][0] == 0:
            excel = self.excel
            gl = self.grey_level
            d = self.distance
            id = self.imglist[index].split('_')[1]
            if self.imglist[index].split('_')[0] == 'M':
                ma = 1
            else:
                ma = 0
            for i, name in enumerate(excel.Chart.array):
                if str(name) == id:
                    swe = excel.SWE.array[i]
                    img = Image.open(self.imgdir + self.imglist[index])
                    img = img.convert('L')
                    npimg = np.array(img)
                    intensity = npimg.sum() / (npimg.shape[0] * npimg.shape[1])
                    qimg = np.array(img.quantize(gl))

                    glcm0 = greycomatrix(qimg, distances=[d], angles=[0], levels=gl, symmetric=True, normed=True)
                    glcm45 = greycomatrix(qimg, distances=[d], angles=[np.pi / 4], levels=gl, symmetric=True,
                                          normed=True)
                    glcm90 = greycomatrix(qimg, distances=[d], angles=[np.pi / 2], levels=gl, symmetric=True,
                                          normed=True)
                    glcm135 = greycomatrix(qimg, distances=[d], angles=[3 * np.pi / 4], levels=gl, symmetric=True,
                                           normed=True)

                    feather_list = [self.get_feather(glcm0, gl),
                                    self.get_feather(glcm45, gl),
                                    self.get_feather(glcm90, gl),
                                    self.get_feather(glcm135, gl)]

            input_data = [swe, intensity] + feather_list[0] + feather_list[1] + feather_list[2] + feather_list[3]
            input_data = np.array(input_data, dtype='float32')
            self.datalist[index] = input_data
            intensor = torch.from_numpy(input_data)
            label = np.array(ma, dtype='int')
            self.labelist[index] = label
            label_tensor = torch.from_numpy(label)
        else:
            intensor = torch.from_numpy(self.datalist[index])
            label_tensor = torch.from_numpy(np.array(self.labelist[index], dtype='int'))

        return intensor, label_tensor

    def __len__(self):
        return len(self.imglist)

    def get_feather(self, glcm, gray_level):
        Ent = 0.0
        Idm = 0.0
        for i in range(gray_level):
            for j in range(gray_level):
                Idm += glcm[i][j] / (1 + (i - j) * (i - j))
                if glcm[i][j] > 0.0:
                    Ent += glcm[i][j] * math.log(glcm[i][j])
        return [Ent[0][0], Idm[0][0],
                greycoprops(glcm, prop='contrast')[0][0],
                greycoprops(glcm, prop='dissimilarity')[0][0],
                greycoprops(glcm, prop='homogeneity')[0][0],
                greycoprops(glcm, prop='ASM')[0][0],
                greycoprops(glcm, prop='energy')[0][0],
                greycoprops(glcm, prop='correlation')[0][0]]
