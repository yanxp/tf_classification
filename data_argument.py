import tensorflow as tf
import numpy as np
import math
import cv2
import sys
import os
from scipy import ndimage
import random
from PIL import Image
###########################################################################################
training_index = './train.txt'
newlabel_index ='./newlabel.txt'
classnum=17
maxImageNum=1360
#############################################################################################

def read_traing_list():
	train_image_dir = []
	train_label_dir = []
	reader = open(training_index)
	while 1:
		line = reader.readline()
		tmp = line.split(" ")
		if not line:
			break
		train_image_dir.append(tmp[0])
		train_label_dir.append(tmp[1][0:-1])
	# print train_image_dir[1:maxImageNum]
	# print train_label_dir[1:maxImageNum]
	reader.close()
	return train_image_dir, train_label_dir
def distort_image():
	train_image_dir, train_label_dir = read_traing_list()
	label_reader = open(newlabel_index,"w")
	for idx in range(len(train_image_dir)):
		print("idx:",idx)
	# ########## flip_left_rightand rotate ###################
		image_path = "17flowers/jpg/"+str(train_label_dir[idx])+"/"+str(train_image_dir[idx])
		im=Image.open(image_path)
	# 	width,height=im.size
	# 	# print (image_path, height,width)
	# 	box_left_up = (0, 0, 227, 227)
	# 	box_right_up=(width-227,0,width,227)
	# 	box_left_down=(0,height-227,227,height)
	# 	box_right_down=(width-227,height-227,width,height)
	# 	center_x=width/2
	# 	center_y=height/2
	# 	box_center=(center_x-227/2,center_y-227/2,center_x+227/2,center_y+227/2)
	# 	box_left_up_img = im.crop(box_left_up)
	# 	box_right_up_img = im.crop(box_right_up)
	# 	box_left_down_img = im.crop(box_left_down)
	# 	box_right_down_img = im.crop(box_right_down)
	# 	box_center_img = im.crop(box_center)
	# 	box_left_up_path = image_path[:-4]+"_left_up.png"
	# 	box_right_up_path = image_path[:-4]+"_right_up.png"
	# 	box_left_down_path = image_path[:-4]+"_left_down.png"
	# 	box_right_down_path = image_path[:-4]+"_right_down.png"
	# 	box_center_path = image_path[:-4]+"_center.png"
	# 	box_left_up_img.save(box_left_up_path)
	# 	box_right_up_img.save(box_right_up_path)
	# 	box_left_down_img.save(box_left_down_path)
	# 	box_right_down_img.save(box_right_down_path)
	# 	box_center_img.save(box_center_path)
	# 	label_reader.write(box_left_up_path+" "+str(train_label_dir[idx])+"\n")
	# 	label_reader.write(box_right_up_path+" "+str(train_label_dir[idx])+"\n")
	# 	label_reader.write(box_left_down_path+" "+str(train_label_dir[idx])+"\n")
	# 	label_reader.write(box_right_down_path+" "+str(train_label_dir[idx])+"\n")
	# 	label_reader.write(box_center_path+" "+str(train_label_dir[idx])+"\n")
	# 	os.system("rm ")
	# label_reader.close()
		# raw_input()
		# resize_img = im.resize((227, 227)) 
	# 	#rotate_45 = im.rotate(45) 
	# 	flip_horizon = im.transpose(Image.FLIP_LEFT_RIGHT) 
	# 	#flip_vertical = im.transpose(Image.FLIP_TOP_BOTTOM)
	# 	#rotate_90 = im.transpose(Image.ROTATE_90) 
	# 	#rotate_180 = im.transpose(Image.ROTATE_180) 
	# 	#rotate_270 = im.transpose(Image.ROTATE_270) 
	# 	flip_resize_path = image_path[:-4]+"_0.png"
	# 	resize_img.save(flip_resize_path)
	# 	label_reader.write(flip_resize_path+" "+str(train_label_dir[idx])+"\n")
	# label_reader.close()
	############## change brightness #############
	# 	image_path = "17flowers/jpg/"+str(train_label_dir[idx])+"/"+str(train_image_dir[idx])
	# 	image_tmp = cv2.imread(image_path,cv2.IMREAD_COLOR) 
	# 	brightness_image=image_tmp
	# 	for i in range(image_tmp.shape[0]):
 # 			for j in range(image_tmp.shape[1]):
 # 		  		brightness_image[i,j] = (255-image_tmp[i,j,0],255-image_tmp[i,j,1],255-image_tmp[i,j,2])
 # 		brightness_image_path = image_path[:-4]+"_4.png"
 #  		cv2.imwrite(brightness_image_path,brightness_image)  
	# 	label_reader.write(brightness_image_path+" "+str(train_label_dir[idx])+"\n")
	# label_reader.close()
if __name__=="__main__":
    distort_image()