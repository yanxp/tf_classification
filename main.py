import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from PIL import Image
from alexnet import AlexNet
cwd = os.getcwd()+"/17flowers/jpg/"
classes={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_and_encode(filename):
    writer = tf.python_io.TFRecordWriter(filename)
    file=open("train.txt",'w')
    test_file=open("test.txt",'w')
    for index, name in enumerate(classes):
        class_path = cwd + str(name) + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            print img_path
            file.write(img_name+" "+str(index)+"\n")
            test_file.write(img_name+" "+str(index)+"\n")
            img = cv2.imread(img_path)
            img=cv2.resize(img,(227,227),interpolation=cv2.INTER_CUBIC)
            rows=img.shape[0]
            cols=img.shape[1]
            depth=img.shape[2]
            img_raw = img.tostring()   
            example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(index)),
            'image': _bytes_feature(img_raw)
             }))
        writer.write(example.SerializeToString())
    writer.close()
    file.close()
    test_file.close()

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
    image = tf.decode_raw(features['image'], tf.uint8)
    image=tf.reshape(image,[227,227,3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label

filename="train.tfrecords"
# write_and_encode(filename)
image, label = read_and_decode(filename)
# # # print (label,image)
# sess = tf.Session()

# # Required. See below for explanation
# init = tf.initialize_all_variables()
# sess.run(init)
# tf.train.start_queue_runners(sess=sess)

# # grab examples back.
# # first example from file
# image_val_1, label_val_1 = sess.run([image, label])
# # second example from file
# image_val_2, label_val_2 = sess.run([image, label])
# print (image_val_2.shape,label_val_2)
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=256,
    capacity=2000,
    min_after_dequeue=1000)
# print images_batch.shape,labels_batch.shape

# net = inference({'data': images_batch})

# y_pred = net.layers['fc8']
# pred = tf.nn.softmax(y_pred)
# simple model
# images_batch=tf.reshape(images_batch,[-1,227*227*3])
# w = tf.get_variable("w1", [227*227*3, 17])
# y_pred = tf.matmul(images_batch, w)
train_layers = ['fc8', 'fc7', 'fc6']
num_classes = 17
model = AlexNet(images_batch,0.5, num_classes, train_layers)
y_pred = model.fc8

labels_batch=tf.one_hot(labels_batch,17)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

# for monitoring*
loss_mean = tf.reduce_mean(loss)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mean)

correct=tf.equal(tf.argmax(y_pred,1),tf.argmax(labels_batch,1))

train_precision=tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    model.load_initial_weights(sess)
    # sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for i in xrange(10000):
      # pass it in through the feed_dict
      _, loss_val,precision= sess.run([train_op, loss_mean,train_precision])
      print loss_val,precision