import cv2
import numpy as np
import math
import scipy.io as sio
import tensorflow as tf

dataset = tf.contrib.data.Dataset
'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.

'''



#===========LABEL VISUALIZER==============

font_scale = 0.5
font_Face = cv2.FONT_HERSHEY_SIMPLEX

#rgb
label_to_colours = []

label_to_texts = []

def get_label_max_length(label, texts):
    size = len(label)
    maxsz=0
    for k in texts:
        maxsz = len(k)

    return size, maxsz


def center_text(img, text, color, rect):
    text = text.strip()
    font = font_Face
    textsz = cv2.getTextSize(str(text), font, font_scale, 1)[0]

    textX = (rect.w-textsz[0]) / 2 + rect.x
    textY = (rect.h+textsz[1]) / 2 + rect.y

    cv2.putText(img, text, (textX, textY), font, font_scale, [0, 0, 0], 2)
    cv2.putText(img, text, (textX, textY), font, font_scale, color, 1)
    return img

def draw_label(label, texts):
    size, maxsz = get_label_max_length(label, texts)

    column = min(size, 15)
    row = int(math.ceil(size/15.0))

    wnd_w = (maxsz*10+50)
    wnd_h = 30
    width = row * wnd_w
    height = column * wnd_h

    img = np.zeros((height, width, 3), np.uint8)
    
    class rect:
        def __init__(self, w, h, colmax, rowmax):
            self.x=0
            self.y=0
            self.w=w
            self.h=h
            self.cmax = colmax
            self.rmax = rowmax
            self.c=0
            self.r=0
        def next(self):
            self.y += self.h
            self.c += 1
            if self.c == self.cmax:
               self.x += self.w
               self.r += 1
               self.c = 0
               self.y = 0 
            

    r = rect(wnd_w, wnd_h, column, row)

    for color, text in zip(label, texts):
        
        color = color.astype(dtype=int).tolist()[::-1]
        cv2.rectangle(img, (r.x, r.y), (r.x+r.w, r.y+r.h), color ,-1)
        img = center_text(img, text, [255, 255, 255], r)
        r.next()

    return img


def draw_labels(seg_img):
    color_idx = np.unique(seg_img).astype(int)
    color_list = []
    text_list = []
    for i in color_idx:
        color_list.append(label_to_colours[i])
        text_list.append(label_to_texts[i])

    img = draw_label(color_list, text_list)
    return img


def draw_anno(seg_img):
    shape = seg_img.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            img[i, j, :] = label_to_colours[seg_img[i, j]][::-1]
    return img

def draw_anno_label(seg_img):
    label = draw_labels(seg_img)
    anno = draw_anno(seg_img)
    return anno, label

#==========OBSTACLE FUNCTION=========
def output_label(seg_img, path):
    img = draw_labels(seg_img)
    cv2.imwrite(path, img)

def anno_to_color(seg_img):
    return draw_anno(seg_img)

def output_anno_label(seg_img, path):
    img = anno_to_color(seg_img)
    cv2.imwrite(path, img)
    name, ext = path.rsplit('.', 1)
    path = name + '_label.' + ext
    print(path)
    output_label(seg_img, path)


def output_all_label(path):
    img = draw_label(label_to_colours, label_to_texts)
    cv2.imwrite(path, img)


def load_label(mat_file):
    import scipy.io as sio
    mat = sio.loadmat(mat_file)
    global label_to_colours
    global label_to_texts
    label_to_colours = mat['colors']
    label_to_texts = mat['names']
    return label_to_colours, label_to_texts

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='draw some labels')
    parser.add_argument('--mat', type=str, default='', help='a .mat file which contains \'colors\' and \'names\' columns')
    parser.add_argument('--output', type=str, default='./labels.png', help='output path of label image')
    
    
    args = parser.parse_args()

    assert args.mat != ''
    assert args.output != ''
    
    import os

    input_path = os.path.abspath(args.mat)
    output_path = os.path.abspath(args.output)

    load_label(input_path)
    output_all_label(output_path)
    print('Save label image to: {}'.format(output_path))


