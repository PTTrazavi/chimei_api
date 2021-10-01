# web app setup
from flask import Flask, request, render_template, send_file, jsonify
import base64

# import some common libraries
import os
import tensorflow as tf
print("tensorflow version:", tf.__version__)
from tensorflow.python.keras.backend import set_session
import keras
import segmentation_models as sm

from utils import dcm_to_imgs, find_max_min, calculate_EF, make_donut, classification, convert_frames_to_video

# Set path
folder_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # <app> folder
template_dir = os.path.join(folder_path, "html", "templates")
static_dir = os.path.join(folder_path, "html", "static")
media_dir = os.path.join(folder_path, "media")
model_path = os.path.join(folder_path, "model", "UNET_4626_1to40_jf_26_0.92_model.h5")
model_class_path = os.path.join(folder_path, "model", "custom_4_classes_58_0.87.h5")

testdata_root = os.path.join(media_dir, "dicom_source")
savedata_root = os.path.join(media_dir, "dicom_result")

app = Flask(__name__, template_folder=template_dir, static_url_path='/static', static_folder=static_dir)

@app.route('/chimei_api', methods=['POST'])
def run_inference():
    json_list = []

    if request.method == 'POST':
        if 'filename' not in request.files:
            json_list = [{'image': 'there is no filename in form!'}]
        dcm_file = request.files['filename']
        # mkdir
        folder_name = dcm_file.filename.split('.')[0]
        if not os.path.exists(os.path.join(testdata_root, folder_name)):
            os.mkdir(os.path.join(testdata_root, folder_name))
        if not os.path.exists(os.path.join(savedata_root, folder_name)):
            os.mkdir(os.path.join(savedata_root, folder_name))
        testdata_folder = os.path.join(testdata_root, folder_name)
        savedata_folder = os.path.join(savedata_root, folder_name)
        # get images from dicom file
        images = dcm_to_imgs(dcm_file, dcm_file.filename, testdata_folder)
        print("dicom contains", len(images), "images!")
        # convert to mp4
        convert_frames_to_video(testdata_folder, static_dir, folder_name)
        print("converted to mp4 video!")
        # check image quantity
        testdatalist = os.listdir(testdata_folder)
        print("total test images:", len(testdatalist))
        savedatalist = os.listdir(savedata_folder)
        print("total predicted images (before):", len(savedatalist))
        # sort the file name
        testdatalist.sort()
        # segmentation and get the max min images
        max_key, min_key, max_file, min_file, heart_max, heart_min, test_dict = find_max_min(testdatalist,
                            testdata_folder, savedata_folder, media_dir, folder_name,
                            preprocess_input, gSess, gGraph, gModel)

        # check how many masks were predicted
        savedatalist = os.listdir(savedata_folder)
        print("total predicted images (after):", len(savedatalist))

        # calculate EF
        EF = calculate_EF(max_key, min_key, heart_max, heart_min,
                            preprocess_input, gSess, gGraph, gModel,
                            testdata_folder, savedata_folder, test_dict, media_dir, folder_name)

        # make donut image
        make_donut(max_file, min_file, testdata_folder, media_dir, folder_name,
                        preprocess_input, gSess, gGraph, gModel)

        # RWMA model prediction
        RWMA = classification(os.path.join(media_dir, folder_name+"_donut.png"),
                                preprocess_input, gSess, gGraph, labels, gGap_weights,
                                gModel_class_cam, media_dir, folder_name)

        # generate json response
        with open(os.path.join(media_dir, folder_name+"_end_diastole.jpg"), 'rb') as diastole:
            diastole_string = base64.b64encode(diastole.read())

        with open(os.path.join(media_dir, folder_name+"_end_systole.jpg"), 'rb') as systole:
            systole_string = base64.b64encode(systole.read())

        with open(os.path.join(media_dir, folder_name+"_cam.png"), 'rb') as cam:
            cam_string = base64.b64encode(cam.read())


    json_list = [{'dicom_mp4': "/static/" + folder_name + "/dicom.mp4",
                  'end_diastole': diastole_string.decode('utf-8'),
                  'end_systole': systole_string.decode('utf-8'),
                  'cam': cam_string.decode('utf-8'),
                  'EF': str(EF),
                  'RWMA': RWMA}]

    return jsonify(json_list)

@app.route('/', methods=['GET'])
def run_app():
    return render_template('index.html')

if __name__ == "__main__":
    # Load segmentation model
    BACKBONE = 'efficientnetb4'
    CLASSES = ['bg', 'heart']
    preprocess_input = sm.get_preprocessing(BACKBONE)
    n_classes = len(CLASSES) + 1 # add unlabelled
    activation = 'softmax'
    ### use session and graph for django!!!
    gSess = tf.Session()
    gGraph = tf.get_default_graph()
    set_session(gSess)
    gModel = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    gModel.load_weights(model_path)
    print("###############")
    print("segmentation model loaded!!!")
    print("###############")

    # load classification model
    labels = {0: 'ApicalAnterior', 1: 'Basal', 2: 'Septal', 3: 'normal'}
    gModel_class = tf.keras.models.load_model(model_class_path)
    # CAM model
    gGap_weights = gModel_class.layers[-1].get_weights()[0]
    gModel_class_cam = tf.keras.models.Model(
                        inputs=gModel_class.input,
                        outputs=(gModel_class.layers[-3].output,gModel_class.layers[-1].output))
    print("###############")
    print("classification model loaded!!!")
    print("###############")

    # app.run(debug=True, host='0.0.0.0', port=15002, use_reloader=False, ssl_context='adhoc')
    # app.run(debug=True, host='0.0.0.0', port=15002, use_reloader=False, ssl_context=(os.path.join(folder_path, "html/ssl/openaifab.com/fullchain3.pem"), os.path.join(folder_path, "html/ssl/openaifab.com/privkey3.pem")))
    app.run(debug=True, host='0.0.0.0', port=15002, use_reloader=False) # use_reloader=False
