from yolo3_one_file_to_detect_them_all import *

# define the model
model = make_yolov3_model()
# load model weights into memory in a format that we can set into our keras model
weight_reader = WeightReader('yolov3.weights')
# set the model weights into the model
weight_reader.load_weights(model)

# save the model to file
model.save('model.h5')


