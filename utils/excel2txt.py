import os
import time
import math
import threading
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.feature import greycomatrix


def split_list(listTemp, n):
    listsplit = []
    for i in range(0, len(listTemp), int(len(listTemp) / n)):
        listsplit.append(listTemp[i:i + int(len(listTemp) / n)])

    return listsplit


def feature_computer(p, gray_level):
    Con = 0.0
    Ent = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Ent += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Ent, Idm


def getglcm(img, d, gl):
    crop_qimg = np.array(img.quantize(gl))
    # h, w = qimg.shape
    # crop_qimg = qimg[h // 5:h - h // 5, w // 5:w - w // 5]

    glcm0 = greycomatrix(crop_qimg, distances=[d], angles=[0], levels=gl, symmetric=True, normed=True)
    glcm45 = greycomatrix(crop_qimg, distances=[d], angles=[45], levels=gl, symmetric=True, normed=True)
    glcm90 = greycomatrix(crop_qimg, distances=[d], angles=[90], levels=gl, symmetric=True, normed=True)
    glcm135 = greycomatrix(crop_qimg, distances=[d], angles=[135], levels=gl, symmetric=True, normed=True)

    asm0, con0, ent0, idm0 = feature_computer(glcm0, gl)
    asm45, con45, ent45, idm45 = feature_computer(glcm45, gl)
    asm90, con90, ent90, idm90 = feature_computer(glcm90, gl)
    asm135, con135, ent135, idm135 = feature_computer(glcm135, gl)

    avg_asm = np.squeeze((asm0 + asm45 + asm90 + asm135) / 4)
    avg_con = np.squeeze((con0 + con45 + con90 + con135) / 4)
    avg_ent = np.squeeze((ent0 + ent45 + ent90 + ent135) / 4)
    avg_idm = np.squeeze((idm0 + idm45 + idm90 + idm135) / 4)

    return avg_asm, avg_con, avg_ent, avg_idm


def job(imglist):
    global txtlist
    global miss
    for imname in tqdm(imglist):
        id = imname.split('_')[1]
        if imname.split('_')[0] == 'M':
            ma = 1
        else:
            ma = 0
        for i, name in enumerate(excel.Chart.array):
            if str(name) == id:
                age = excel.Age.array[i]
                swe = excel.SWE.array[i]
                ssize = excel.ssize.array[i]
                lsize = excel.lsize.array[i]
                area = (ssize / 2) * (lsize / 2) * math.pi
                scale = ssize / lsize

                img = Image.open(imgdir + imname)
                img = img.convert('L')
                cimg = np.array(img)
                # h, w = cimg.shape
                # cimg = cimg[h // 5:h - h // 5, w // 5:w - w // 5]
                intensity = cimg.sum() / (cimg.shape[0] * cimg.shape[1])

                asm0, con0, ent0, idm0 = getglcm(img, 2, 32)
                asm1, con1, ent1, idm1 = getglcm(img, 1, 32)
                asm2, con2, ent2, idm2 = getglcm(img, 1, 64)

                txtlist.append(' '.join(np.array([ma, age, swe, area, scale, intensity,
                                                  asm0, con0, ent0, idm0,
                                                  asm1, con1, ent1, idm1,
                                                  asm2, con2, ent2, idm2,
                                                  imgdir + imname + '\n'], dtype='str')))
                time.sleep(0.5)
                break
            if i == 143:
                miss += 1


if __name__ == '__main__':
    global txtlist
    global miss
    exceldir = '/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/2018LN.xlsx'
    imgdir = '/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/train_seg_cont/'
    txtdir = '/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/train_seg_cont.txt'
    excel = pd.read_excel(exceldir)
    imglist = os.listdir(imgdir)
    imglist = split_list(imglist, 6)
    txtlist = []
    miss = 0
    t1 = threading.Thread(target=job, args=(imglist[0],))
    t2 = threading.Thread(target=job, args=(imglist[1],))
    t3 = threading.Thread(target=job, args=(imglist[2],))
    t4 = threading.Thread(target=job, args=(imglist[3],))
    t5 = threading.Thread(target=job, args=(imglist[4],))
    t6 = threading.Thread(target=job, args=(imglist[5],))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()

    f = open(txtdir, 'w')
    f.writelines(txtlist)
    f.close()
    print('get :', len(txtlist))
    print('miss :', miss)
