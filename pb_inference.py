import tensorflow as tf  # Default graph is initialized when the library is imported
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import pickle
import time
import csv
import sys
import os
import inspect

from inference_utils.image_utils import preprocess_image, resize_image

IMAGE_PATH = sys.argv[1]

# Image processing
image = cv2.imread(IMAGE_PATH)
image = preprocess_image(image)
image, scale = resize_image(image)

# Model File
pb_file = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),'model_files/object_detector.pb')
print(pb_file)

# Label File
label_file = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),'model_files/labels.pickle')
print(label_file)
with open(label_file, 'rb') as handle:
    labels = pickle.load(handle)

try:
    os.remove("output_files/labels_list.csv")
except OSError:
    pass

start_time = time.time()
# Reference: https://github.com/MarvinTeichmann/KittiSeg/issues/113
with tf.Graph().as_default() as graph:  # Set default graph as graph

    with tf.Session() as sess:
        # Load the graph in graph_def
        print("load graph")

        # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
        with gfile.FastGFile(pb_file, 'rb') as f:

            # Set FCN graph to the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            # Import a graph_def into the current default Graph
            # (In this case, the weights are (typically) embedded in the graph)
            tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="",
                    op_dict=None,
                    producer_op_list=None
                    )

            # initialize_all_variables
            tf.global_variables_initializer()

            # INFERENCE Here
            l_input = graph.get_tensor_by_name('input_1:0')  # Input Tensor
            l_output = graph.get_tensor_by_name('non_maximum_suppression_1/ExpandDims:0')  # Output Tensor

            detections = sess.run(l_output, feed_dict={l_input: np.expand_dims(image, axis=0)})

            # compute predicted labels and scores
            predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
            scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

            # Write output labels and locations
            height, width, _ = image.shape
            predictions_locs = []
            for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
                if score < 0.3:
                    continue
                b = detections[0, idx, :4].astype(int)
                predictions_locs.append([labels[label],
                                         float(b[0]) / float(width),  # x1
                                         float(b[1]) / float(height),  # y1
                                         float(b[2]) / float(width),  # x2
                                         float(b[3]) / float(height),  # y2
                                         ])


output_files = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),'output_files/labels_list.csv')
with open(output_files, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(predictions_locs)

print(time.time() - start_time)
