import os
import numpy as np
import cv2
import multiprocessing
import time

partnum=-576

directory = raw_input('what directory do you want to parse? ' )
savelocation = raw_input('what directory do you want to save to? ')
def cutphoto( image, save, number):
    img=cv2.imread(image)
    print ('parsing image ' + str(image))
    for y in range(0,18):
        for z in range(0,32):
            part=img[y*60:(y+1)*60,z*60:(z+1)*60]
            number=number+1
            cv2.imwrite(save + '/' + str(number) + '.jpg', part)
    print ('thread ' + str(image) +' completed')

if __name__ == '__main__':
    jobs =[]
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            image=os.path.join(subdir, file)
            partnum=partnum+576
            #cutphoto(image, savelocation, partnum)
            p=multiprocessing.Process(target=cutphoto, args=(image, savelocation, partnum,))
            jobs.append(p)
            p.start()