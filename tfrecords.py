import tensorflow as tf
import numpy as np
# import cv2
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
            img = img.resize((227, 227))
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
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

# if __name__ == '__main__':
#     filename="train.tfrecords"
#     # write_and_encode(filename)
#     img, label = read_and_decode(filename)
#     print (img.shape,label.shape)
#     raw_input()

#     img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                     batch_size=1, capacity=2000,
#                                                     min_after_dequeue=1000)
  
#     init = tf.initialize_all_variables()

#     with tf.Session() as sess:
#         sess.run(init)
  
#         threads = tf.train.start_queue_runners(sess=sess)
#         for i in range(3):
#             val, l= sess.run([img_batch, label_batch])
#             # l = to_categorical(l, 17)
            # print(val, l)
filename="train.tfrecords"
    # write_and_encode(filename)
image, label = read_and_decode(filename)
sess = tf.Session()

# Required. See below for explanation
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# grab examples back.
# first example from file
image_val_1, label_val_1 = sess.run([image, label])
# # second example from file
# label_val_2, image_val_2 = sess.run([image, label])
print (image_val_1.shape,label_val_1.shape)