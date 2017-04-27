# %%
import argparse

import numpy as np
import skimage.io
import skimage.draw
from skimage.transform import resize
import chainer
from chainer import serializers
import cv2
import ssd_net
import draw

print("a")
parser = argparse.ArgumentParser(
    description='detect object')
#parser.add_argument('path', help='Path to image file')
parser.add_argument('--name', default="", help='Path to image file to save')

args = parser.parse_args()

mean = np.array([104, 117, 123])
#load image
capture = cv2.VideoCapture(0)
    
if capture.isOpened() is False:
	raise("IO Error")


#load SSD model
model = ssd_net.SSD()
serializers.load_npz("model/ssd.model", model)

while True:
	ret, image = capture.read()
	if ret == False:
		continue
	

#image = skimage.img_as_float(skimage.io.imread(args.path, as_grey=False)).astype(np.float32)

	img = resize(image, (300, 300))
	img = img*255 - mean[::-1]
	img = img.transpose(2, 0, 1)[::-1]

	#change shape for imputting in chainer
	x = chainer.Variable(np.array([img], dtype=np.float32))
	model(x, 1, 1, 1, 1)

	prior = model.mbox_prior.astype(np.float32)
	loc = model.mbox_loc.data[0]
	conf = model.mbox_conf_softmax_reahpe.data[0]
	cand = draw.detect(prior, loc, conf)
	draw.draw(image, cand, args.name)

cv2.destroyAllWindows()