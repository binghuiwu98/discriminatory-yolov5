import os
import shutil

imgpath = 'images'
cwd = os.getcwd()
source_txt = ['./ImageSets/Main/train.txt', './ImageSets/Main/val.txt', './ImageSets/Main/test.txt']
destination_img = ['./Train', './Val', './Test']
source_img = os.path.join(cwd,imgpath)

for so, de in zip(source_txt, destination_img):
    f = open(so)
    filenamelist = []
    line = f.readline()
    while line:
        name = line[:-1]
        filenamelist.append(name)
        line = f.readline()
    f.close()
    # print(filenamelist)
    for filename in filenamelist:
        jpgpath = os.path.join(source_img, filename + '.jpg')
# makedir
        if os.path.exists(jpgpath):
            jpgdest = os.path.join(de, filename + '.jpg')
            if os.path.exists(jpgdest):
                print(jpgdest + " already exists.")
            else:
                shutil.copy(jpgpath, jpgdest)
                print(jpgdest + " copied successfully.")
        else:
            print(jpgpath + " not exists.")

