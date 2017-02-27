import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from PIL import Image

cwd = os.getcwd()+"/17flowers/jpg/"
classes={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
def write_and_encode(filename):
    writer = tf.python_io.TFRecordWriter(filename)
    file=open("train.txt",'w')
    for index, name in enumerate(classes):
        class_path = cwd + str(name) + "/"
        # print ("class_path",class_path)
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            file.write(img_name+" "+str(index)+"\n")
            # print ("img_path",img_path)
            img = Image.open(img_path)
            img = img.resize((224, 224))############################################################################################  
#!/usr/bin/python2.7  
# -*- coding: utf-8 -*-  
#Author  : zhaoqinghui  
#Date    : 2016.5.11  
#Function: add image  
##########################################################################################
import tensorflow as tf
import numpy as np
import math
import cv2
import sys
import os
from scipy import ndimage
import random

###########################################################################################
#设置自己的参数
###########################################################################################
training_index = './traini.txt'
newlabel_index ='./newlabel.txt'
classnum=36
maxImageNum=360
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
    #print train_image_dir[1:maxImageNum]
    #print train_label_dir[1:maxImageNum]
    reader.close()
    return train_image_dir, train_label_dir

def distort_image():
    train_image_dir, train_label_dir = read_traing_list()
    label_reader = open(newlabel_index,"w")
    for idx in range(len(train_image_dir)):
        image_path = str(train_image_dir[idx])
        image_tmp = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        rotate_image = ndimage.rotate(image_tmp,random.randint(7,30))
        rotate_image = cv2.resize(rotate_image,(28,28))
        rotate_image_path = image_path[:-4]+"_1.png"
        print rotate_image_path
        cv2.imwrite(rotate_image_path,rotate_image)
        rotate_image2 = ndimage.rotate(image_tmp,random.randint(330,355))
        rotate_image2 = cv2.resize(rotate_image2,(28,28))
        rotate_image_path2 = image_path[:-4]+"_2.png"
        cv2.imwrite(rotate_image_path2,rotate_image2)
        label_reader.write(rotate_image_path+" "+str(train_label_dir[idx])+"\n")
        label_reader.write(rotate_image_path2+" "+str(train_label_dir[idx])+"\n")
    label_reader.close()
    print "done"
if __name__="__main__":
        distort_image()


            img_raw = img.tobytes()   
            # print ("index:",index)           
            example = tf.train.Example(features=tf.train.Features(feature={
              "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
             }))
            writer.write(example.SerializeToString())
    writer.close()
    file.close()
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

if __name__ == '__main__':
    filename="train.tfrecords"
    write_and_encode(filename)
    img, label = read_and_decode(filename)
    # print (img,label)
    # raw_input()

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
  
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
  
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l= sess.run([img_batch, label_batch])
            #l = to_categorical(l, 12)
            print(val.shape, l)