from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from pyexcel_ods import save_data
from collections import OrderedDict

data = OrderedDict()
sheet = []
img_name=[]

RESIZE_FINAL = 227
GENDER_LIST =['m','f']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('model_sex_dir', '',
                           'Model directory (where sex model lives)')

tf.app.flags.DEFINE_string('class_type', 'age',
                           'Classification type (age|gender)')


tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '',
                           'File (Image) or File list (Text/No header TSV) to process')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

FLAGS = tf.app.flags.FLAGS

def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)
        
        return [row[0] for row in reader]

class ImportGraph(object):
    def __init__(self, model_dir, class_type):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(allow_soft_placement=True))
        with self.graph.as_default():
            # import saved model from loc into local graph
            with tf.device('/device:GPU:0'):
                label_list = AGE_LIST if class_type == 'age' else GENDER_LIST
                nlabels = len(label_list)
                model_fn = select_model(FLAGS.model_type)
                self.images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                logits = model_fn(nlabels, self.images, 1, False)
                saver = tf.train.Saver()
                requested_step = FLAGS.requested_step if FLAGS.requested_step else None
                checkpoint_path = '%s' % model_dir
                model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
                saver.restore(self.sess, model_checkpoint_path)
                self.softmax_output = tf.nn.softmax(logits)

    def run(self, data):
        with tf.Session().as_default():
            data = data.eval()
        return self.sess.run(self.softmax_output, feed_dict={self.images: data})

def classify(cl_model, label_list, coder, image_file, writer, c_type):
    try:
        # print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)
        batch_results = cl_model.run(image_batch)
        output = batch_results[0]
        batch_sz = batch_results.shape[0]

        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
        dt = [image_file,str(label_list[best]),str(output[best])]
        sheet.append(dt)
        # nlabels = len(label_list)
        # if nlabels > 2:
        #     output[best] = 0
        #     second_best = np.argmax(output)
        #     print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

        if writer is not None:
            writer.writerow((image_file, best_choice[0], '%.2f' % best_choice[1]))
        if c_type == 1:
            return "age: %s prob: %.2f" % best_choice
        else:
            return "sex: %s prob: %.2f" % best_choice

    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)

def main(argv=None):  # pylint: disable=unused-argument

    files = []
    
    if FLAGS.face_detection_model:
        print('Using face detector (%s) %s' % (FLAGS.face_detection_type, FLAGS.face_detection_model))
        face_detect = face_detection_model(FLAGS.face_detection_type, FLAGS.face_detection_model)
        face_files, rectangles = face_detect.run(FLAGS.filename)
        print(face_files)
        files += face_files

    model_age = ImportGraph(FLAGS.model_dir, "age")
    model_sex = ImportGraph(FLAGS.model_sex_dir, "sex")


    coder = ImageCoder()

    # Support a batch mode if no face detection model
    if (os.path.isdir(FLAGS.filename)):
        for relpath in os.listdir(FLAGS.filename):
            abspath = os.path.join(FLAGS.filename, relpath)
            
            if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                print(abspath)
                files.append(abspath)
    else:
        files.append(FLAGS.filename)
        # If it happens to be a list file, read the list and clobber the files
        if any([FLAGS.filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
            files = list_images(FLAGS.filename)
        
    writer = None
    output = None
    if FLAGS.target:
        print('Creating output file %s' % FLAGS.target)
        output = open(FLAGS.target, 'w')
        writer = csv.writer(output)
        writer.writerow(('file', 'label', 'score'))

    image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
    h=0
    for image_file in image_files:
        start_time = time.time()
        age_label = classify(model_age, AGE_LIST, coder, image_file, writer, 1)
        elapsed_time = time.time() - start_time
        print("time: ", elapsed_time)
        h = h+1
        print(h)
    with open('./Adience_tf.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in sheet:
            employee_writer.writerow(i)
        # sex_label = classify(model_sex, GENDER_LIST, coder, image_file, writer, 2)
    #print(image_files)
    # if FLAGS.single_look:
    #     classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer)
    #     data.update({"Sheet 1": sheet })
    #     save_data("./adience_tf.ods",data)

    # else:
    #     n=0
    #     for image_file in image_files:
    #         start_time = time.time()
    #         classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer)
    #         elapsed_time = time.time() - start_time
    #         n=n+1
    #         print(n)
    #         print("time: ", elapsed_time)
    #     data.update({"Sheet 1": sheet })
    #     save_data("./adience_tf.ods",data)

    if output is not None:
        output.close()
        
if __name__ == '__main__':
    tf.app.run()
