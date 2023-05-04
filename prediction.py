from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
import numpy as np
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from yolo3_one_file_to_detect_them_all import *

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

        def get_label(self):
            if self.label == -1:
                self.label = np.argmax(self.classes)
            return self.label
        
        def get_score(self):
            if self.score == -1:
                self.score = self.classes[self.get_label()]
            return self.score
        
#load and prepare an image
def load_image_pixels(filename, shape):
    #load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    #load the image with the required size
    image = load_img(filename, target_size=shape)
    #convert to numpy array
    image = img_to_array(image)
    #scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    #add a dimension so that we have one sample
    image = expand_dims(image,0)
    return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    #enumerate all boxes
    for box in boxes:
    #enumerate all possible labels
        for i in range(len(labels)):
        #check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='white')
		# draw the box
		ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		pyplot.text(x1, y1, label, color='white')
	# show the plot
	pyplot.show()

#load yolov3 model
model = load_model('model.h5')
#define the expected input shape for the model
input_w, input_h = 416, 416
photo_filename = 'nutriboost.png'
#load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

# make prediction
yhat = model.predict(image)
#summarize the shape of the list of arrays
print([a.shape for a in yhat])

anchors = [[116,90,156,198,373,326],[30,61,62,45,59,119],[10,13,16,30,33,23]]
# define the probability threshold for detected objects
class_threshold = 0.6
boxes = list()
for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold,0.5, input_h, input_w)
#correct the sizes of the bounding boxes for the shape of the image
correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

do_nms(boxes, 0.5)
#define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
 "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
 "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
 "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
 "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
 "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
 "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
 "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
 "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
 "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
#summarize what we found
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])
    #draw what we found
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)

