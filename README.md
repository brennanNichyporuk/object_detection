# object_detection
Running Retinanet Object detector on tensorflow using a pretrained pb file. (Ask me for the pb file)

# Dependencies
- Tensorflow (tested with r1.4.0)
- Numpy
- Opencv (tested with 3.3)
- pickle

# Running Inference
1. Download the pretrained model's pb file and place it in model_files.
2. Open [pb_inference.py](https://github.com/roggirg/object_detection/blob/master/pb_inference.py) and change the name of 
the variable 'IMAGE_PATH' to point to the absolute path of your test image.
3. Run the script.

# Output of Inference
The code is setup to output a csv file containing all the labels of the objects in the image
and the corresponding x and y position. The x and y positions are in the form of ratios of width and height.
