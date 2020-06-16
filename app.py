import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from modelutils import ops as utils_ops
from modelutils import label_map_util
from modelutils import visualization_utils as vis_util


from bridge.bridge_manager import BridgeManager
from models.receiveJobs import ReceiveJobs
import cv2
import urllib.request
import pandas as pd

from flask import Flask
from flask import request

app = Flask(__name__)


def init_variables():
    PATH_TO_CKPT = '/home/administrator/Documents/dl/detection/workspace/training_demo/fine_tuned_model_1/frozen_inference_graph.pb'
    PATH_TO_LABELS = '/home/administrator/Documents/dl/detection/workspace/training_demo/training/label_map.pbtxt'
    NUM_CLASSES = 97

    return PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES

def init_graph():
    print("init graph")

    #PATH_TO_CKPT = '/home/administrator/Documents/dl/detection/workspace/training_demo/fine_tuned_model_1/frozen_inference_graph.pb'

    # Path to label map file
    #PATH_TO_LABELS = '/home/administrator/Documents/dl/detection/workspace/training_demo/training/label_map.pbtxt'

    # Path to image
    #PATH_TO_IMAGE = '/home/administrator/Documents/dl/detection/workspace/training_demo/images/test/ImageCooler1_1554122751441.jpg'
    # #PATH_TO_IMAGE = '/home/administrator/Documents/dl/detection/workspace/training_demo/images/Failed-Images/merge/ImageCokeCooler3_1548998330312.jpg'

    # Path to output file
    #PATH_TO_OUTPUT = '/home/administrator/Documents/dl/detection/workspace/output.jpg'

    # # Number of classes the object detector can identify
    #NUM_CLASSES = 97

    PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES = init_variables()

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        session = tf.Session(graph=detection_graph)    
    #detection_graph = det_graph    
    return detection_graph, session, category_index

detection_graph, sess, category_index = init_graph()

@app.route('/upload-image', methods=['POST'])
def upload_image():
    req_data = request.get_json()
    print(req_data["data_url"])
    data_url = req_data["data_url"]
    job_id = req_data["job_id"]
    return process_image(data_url, job_id)

def process_image(data_url, job_id):
    # PATH_TO_CKPT = '/home/administrator/Documents/dl/detection/workspace/training_demo/fine_tuned_model_1/frozen_inference_graph.pb'

    # # Path to label map file
    # PATH_TO_LABELS = '/home/administrator/Documents/dl/detection/workspace/training_demo/training/label_map.pbtxt'

    # # Path to image
    PATH_TO_IMAGE = '/home/administrator/Documents/dl/detection/workspace/training_demo/images/test/ImageCooler1_1554122751441.jpg'
    # #PATH_TO_IMAGE = '/home/administrator/Documents/dl/detection/workspace/training_demo/images/Failed-Images/merge/ImageCokeCooler3_1548998330312.jpg'

    # # Path to output file
    PATH_TO_OUTPUT = '/home/administrator/Documents/dl/detection/workspace/output.jpg'

    # # Number of classes the object detector can identify
    # NUM_CLASSES = 97


    # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

    # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

    # category_index = label_map_util.create_category_index(categories)


    # # Load the Tensorflow model into memory.
    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')

    #     sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    if detection_graph:
        print("Graph Loaded")

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # print(detection_boxes, detection_scores, detection_classes, num_detections)
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value

    # read image from url using CV2
    url_response = urllib.request.urlopen(data_url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, -1)

    rows = image.shape[0]
    cols = image.shape[1]
    inp = cv2.resize(image, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])

    classData = []

    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            print("X {}, y {}, right {}, bottom {}".format(x, y, right, bottom))
            names = list(category_index.values())
            name = 0
            for i in names:
                if i['id'] == classId:
                    name = i['name']
                    break
            if name == 0:
                name = 'not Found'
            classData.append(name)
            print(name + "< == >" + str(classId))
            cv2.rectangle(image, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            text = "{}: {}".format(name, round(score, 4))
            cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    # Save image
    cv2.imwrite(PATH_TO_OUTPUT, image)

    from assets import file_storage
    file_manager = file_storage.FileManager(None)
    file_manager.load_config()
    uploaded_stored_image = file_manager.upload_file(PATH_TO_OUTPUT)
    if upload_image:    
        print(uploaded_stored_image['url'])
        bridge = BridgeManager().get_Instance().get_Bridge()
        # Update ProcessedPath from uploaded_stored_image['url']            
        bridge.get_db().get_session().query(ReceiveJobs).filter_by(id = job_id).update({ReceiveJobs.processedPath:uploaded_stored_image['url']})
        bridge.get_db().get_session().commit()

    df = pd.DataFrame(classData,columns = ['BRAND'])
    df['COUNT']=1
    group_df = df.groupby(['BRAND']).count()[['COUNT']]
    group_df = group_df.reset_index()
    return group_df.to_json(orient ='records')



if __name__ == '__main__':
    init_graph()
    app.run(debug=True,port=8500)