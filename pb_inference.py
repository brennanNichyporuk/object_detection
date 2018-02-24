import tensorflow as tf  # Default graph is initialized when the library is imported
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import pickle
import time
import csv
import sys
import os

from inference_utils.image_utils import preprocess_image, resize_image

IMAGE_PATH = sys.argv[1]

# Image processing
image = cv2.imread(IMAGE_PATH)
image = preprocess_image(image)
image, scale = resize_image(image)

# Model File
pb_file = 'model_files/object_detector.pb'

# Label File
with open('model_files/labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)

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
                x_pos = b[0] + (b[2] - b[0]) / 2
                y_pos = b[1] + (b[3] - b[1]) / 2
                predictions_locs.append([ labels[label],
                                          float(x_pos)/float(width),
                                          float(y_pos)/float(height) ])

try:
    os.remove("output_files/labels_list.csv")
except OSError:
    pass
with open('output_files/labels_list.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(predictions_locs)

print(time.time() - start_time)
