
import cv2
import numpy as np
import glob
import os
import re
import shutil

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

other = []

img_array = []
for filename in glob.glob('./images/*.jpg'):
    other.append(filename)

other.sort(key=natural_keys)

for i in other:
    img = cv2.imread(i)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

shutil.rmtree("images")
os.mkdir("images")
