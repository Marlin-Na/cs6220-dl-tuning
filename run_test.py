#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Jialin Ma
Based on Trevor Goodyear and Raghav Prabhu's script
https://github.gatech.edu/tgoodyear3/cs6220-hw1/blob/master/classify.py
"""
import tensorflow as tf
import sys
import os
import csv

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
Classify images from test folder and predict dog breeds along with score.
'''
def classify_image(image_path, headers):
    f = open('submit.csv','w')
    writer = csv.DictWriter(f, fieldnames = headers)
    writer.writeheader()
    
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("trained_model/retrained_labels.txt")]
   
    # Unpersists graph from file
    with tf.gfile.FastGFile("trained_model/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    files = os.listdir(image_path)
    recs = []
    trueCount = 0.0
    falseCount = 0.0
    with tf.Session() as sess:
         for file in files:
             # Read the image_data
                image_data = tf.gfile.FastGFile(image_path+'/'+file, 'rb').read()
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                predictions = sess.run(softmax_tensor, \
                                       {'DecodeJpeg/contents:0': image_data})

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                records = []
                row_dict = {}
                pathpaths = os.path.split(file)
                row_dict['id'] = pathpaths[-1]
                result = label_lines[top_k[0]] in row_dict['id']
                recs.append([row_dict['id'],label_lines[top_k[0]],result])
                if result:
                    trueCount += 1
                else:
                    falseCount += 1

                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    # print('%s (score = %.5f)' % (human_string, score))
                    row_dict[human_string] = score
                # records.append(row_dict.copy())
                # writer.writerows(records)
    f.close()
    accuracy = trueCount / (trueCount + falseCount)
    print('True: %s False: %s Accuracy: %s' % (trueCount, falseCount, accuracy))

def main():
    test_data_folder = 'test'
    
    template_file = open('sample_submission.csv','r')
    d_reader = csv.DictReader(template_file)

    #get fieldnames from DictReader object and store in list
    headers = d_reader.fieldnames
    template_file.close()
    classify_image(test_data_folder, headers)
    

if __name__ == '__main__':
    main()
