import tensorflow as tf
import os
from enet import ENet, ENet_arg_scope
import numpy as np
slim = tf.contrib.slim
import cv2

from label_loader import *

os.environ['CUDA_VISIBLE_DEVICES']='1'

#########
image_dir = './dataset/testimg/'
images_list = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')])

#########
checkpoint_dir = os.path.abspath("./weights/cityscapes_19")
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

######### 
# DO NOT CHANGE THESE VARIABLES
num_initial_blocks = 1
skip_connections = False
stage_two_repeat = 2
num_classes = 20
blend_org_alpha = 0.5

###################
# input image size
image_width = 640
image_height = 320
#########

label_to_colours, label_to_texts = load_label(os.path.abspath('./labels/cityscapes_19_label.mat'))





'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.
'''

#Create the photo directory
photo_dir = checkpoint_dir + "/test_images"
if not os.path.exists(photo_dir):
    os.mkdir(photo_dir)



def preprocess_image(image):
    image = cv2.resize(image, (image_width, image_height))
    image = image.astype(dtype=np.float32)[np.newaxis, ...] / 255.0
    return image


def output_image(org_image, seg, filename):
    org_shape = org_image.shape
    seg_color = draw_anno(seg)
    seg_color = cv2.resize(seg_color, (org_shape[1], org_shape[0]), interpolation=cv2.INTER_NEAREST)  # resize to original size


    name, ext = os.path.splitext(filename)

    outputname = os.path.join(photo_dir, name+'.png')
    annoname = os.path.join(photo_dir, name+'_anno.png')
    labelname = os.path.join(photo_dir, name+'_label.png')
    blendname = os.path.join(photo_dir, name+'_blend.png')
    print('  -> Saving {}'.format(outputname))
    print('  -> Saving {}'.format(annoname))
    print('  -> Saving {}'.format(labelname))
    print('  -> Saving {}'.format(blendname))


    label = draw_labels(seg)  # draw label list
    blend = org_image * blend_org_alpha + seg_color * (1-blend_org_alpha) # blend with original image

    cv2.imwrite(outputname, org_image)  # output original image
    cv2.imwrite(annoname, seg_color)   # output colored segmentation
    cv2.imwrite(labelname, label)  # output label liste
    cv2.imwrite(blendname, blend)  # output blended image





## initialize tf graph
with tf.Graph().as_default() as graph:
    image_tensors = tf.placeholder(dtype=tf.float32, shape=(1, image_height, image_width, 3), name='input_tensor')
    with slim.arg_scope(ENet_arg_scope()):
        logits, probabilities = ENet(image_tensors,
                                    num_classes=num_classes,
                                    batch_size=1, 
                                    is_training=True,
                                    reuse=None,
                                    num_initial_blocks=num_initial_blocks,
                                    stage_two_repeat=stage_two_repeat,
                                    skip_connections=skip_connections)
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint)

    pred = tf.argmax(probabilities, -1)
    pred = tf.cast(pred, tf.float32)

    sv = tf.train.Supervisor(logdir=None, init_fn=restore_fn)

    try:
        os.makedirs(photo_dir)
    except:
        pass



##### create session
with sv.managed_session() as sess:
    for i in range(len(images_list)):
        org_image = cv2.imread(images_list[i])  # original image

        image = preprocess_image(org_image)   # input image (resized)


        seg = sess.run(pred, feed_dict={image_tensors:image})  # segmentation output (resized)
        seg = seg[0].astype(dtype=int)

        print('Saving image {}/{}'.format(i+1, len(images_list)))
        output_image(org_image, seg, os.path.basename(images_list[i]))





