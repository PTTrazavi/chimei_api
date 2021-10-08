import pydicom
import cv2
import numpy as np
import scipy as sp
import os
import ffmpeg
from PIL import Image
from skimage import io
import random as rng

import tensorflow as tf
import keras
from tensorflow.python.keras.backend import set_session

print("tf gpu device name:", tf.test.gpu_device_name())
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '0'


# dicom to image list function
def dcm_to_imgs(File, fname, testdata_folder):
    print('flie name :',fname)
    if fname[-4:] == '.dcm':
        ds = pydicom.read_file(File)  # read .dcm file
        imgs = ds.pixel_array  # retrieve images
        # save each image in media folder
        for i,j in enumerate(imgs):
            image_fn = '%.5d.jpg' % i
            image_path = os.path.join(testdata_folder, image_fn)
            cv2.imwrite(image_path, j)
        return imgs
    else:
        return 'please upload dicom file'


# images to video function
def convert_frames_to_video(pathIn, static_dir, folder_name, fps=25):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if not f.startswith('.')]
    #for sorting the file names properly
    files.sort()
    for i in files:
        filename = os.path.join(pathIn, i)
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    if not os.path.exists(os.path.join(static_dir, folder_name)):
        os.mkdir(os.path.join(static_dir, folder_name))

    out = cv2.VideoWriter(os.path.join(static_dir, folder_name, "dicom_temp.mp4"),
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, size) # MP42 DIVX mp4v

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    # convert to h264 codec
    process = (
    ffmpeg
    .input(os.path.join(static_dir, folder_name, "dicom_temp.mp4"))
    .output(os.path.join(static_dir, folder_name, "dicom.mp4"),vcodec='libx264')
    .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    out, err = process.communicate()


# save the mask image function
def save_color(pr_mask, pr_mask_path, need_resize=False, target_width=512, target_height=512):
    colors = [(0,0,0),      #bg
              (250,0,0),    #heart
              (0,0,0)]       #unlabel
    height, width = pr_mask.shape[0], pr_mask.shape[1]
    img_mask = Image.new(mode = "RGB", size = (width, height))
    px = img_mask.load()

    for x in range(0,width):
        for y in range(0,height):
             px[x,y] = colors[pr_mask[y][x]]
    # resize if the original size is not 512*512
    if need_resize is True:
        img_mask = img_mask.resize((target_width, target_height), Image.NEAREST)
    # save the mask image
    img_mask.save(pr_mask_path)


# donut image generator
def draw_combined_color(mask_image):
    colors = {0:0,      #bg 0
              1:250,    #diastole 1
              -1:100}    #systole -1

    out_image = np.zeros_like(mask_image)

    for k,v in colors.items():
        class_mask = (mask_image==k) # get the layer of class
        class_mask = class_mask * colors[k]
        out_image = out_image + class_mask

    out_image= out_image.astype(int) # make it int so plt.imshow can show rgb 0-255, float will show 0-1
    return out_image


# get enclosing circle function
def get_enclosing_circle(pr_mask, media_dir, folder_name, threshold=100):
    """
    input:
        pr_mask is numpy array with shape (width, height)
        threshold max 255 default 100
    outputs:
        enclosing circle center and radius
    """
    colors = [0,      #bg
              250,    #heart
              0]      #unlabel

    height, width = pr_mask.shape[0], pr_mask.shape[1]
    img_mask = Image.new(mode = "L", size = (width, height))
    px = img_mask.load()

    for x in range(0,width):
        for y in range(0,height):
             px[x,y] = colors[pr_mask[y][x]]

    img_mask.save(os.path.join(media_dir, folder_name+"_enclosing.jpg"))
    # cv2 process from here
    src = cv2.imread(os.path.join(media_dir, folder_name+"_enclosing.jpg"))
    src_blur = cv2.blur(src, (3,3))
    # cv2.imwrite("enclosing_blur.jpg", src_blur)
    # Detect edges using Canny
    canny_output = cv2.Canny(src_blur, threshold, threshold * 2)
    # cv2.imwrite("enclosing_canny.jpg", canny_output)
    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    # draw result
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # Draw polygonal contour + bonding rects + circles
    temp_i = 0
    temp_r = 0
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        # keep the biggest circle for output
        if int(radius[i]) > temp_r:
          temp_r = int(radius[i])
          temp_i = i
    # cv2.imwrite('enclosing_contours.jpg', drawing)
    return int(centers[temp_i][0]), int(centers[temp_i][1]), int(radius[temp_i])


# Find the max and min images function
def find_max_min(testdatalist, testdata_folder, savedata_folder, media_dir, folder_name,
                    preprocess_input, gSess, gGraph, gModel,
                    need_resize = False, width = 512, height = 512):
    # original images dict
    test_dict = dict(enumerate(testdatalist))
    # count the pixel of heart
    heart_max = 0
    heart_min = width*height
    max_file = str()
    min_file = str()
    max_key = int()
    min_key = int()

    # get the max area image
    for k,i in test_dict.items():
        image = io.imread(os.path.join(testdata_folder, i)) # novel image
        # # check if the size is 512*512
        # convert 1 channel to 3 channel if necessary
        if len(image.shape) == 2:
          image = np.stack((image,)*3, axis=-1)

        image = preprocess_input(image) # use the preprocessing input method based on the backbone you used
        image = np.expand_dims(image, axis = 0)
        # in django/flask predict the model with graph and session
        with gGraph.as_default():
            set_session(gSess)
            pr_mask = gModel.predict(image) # change the shape to (1 H, W)
        pr_mask = pr_mask.squeeze()
        pr_mask = np.argmax(pr_mask, axis = 2)

        save_color(pr_mask, os.path.join(savedata_folder, i.split(".")[0]+".png"), need_resize, width, height)

        # count the pixels of heart(label==1)
        heart = np.sum(pr_mask==1)
        if heart > heart_max:
          heart_max = heart
          max_file = i
          max_key = k

    # get the min area image, the min image should appear after max image
    for k,i in test_dict.items():
        pr_mask = io.imread(os.path.join(savedata_folder, i.split(".")[0]+".png"))
        # count the pixels of heart(label==1)
        heart = np.sum(pr_mask==250)
        if heart < heart_min and k > max_key:
          heart_min = heart
          min_file = i
          min_key = k

    # ########## DEBUG C3929_0.dcm ##########
    # max_key = 121
    # min_key = 127
    # max_file = "00121.jpg"
    # min_file = "00127.jpg"
    # heart_max = 85374
    # heart_min = 61736
    # ###########################

    # check the results
    print("max_key is:", max_key, "max_file is:", max_file, "heart_max is:", heart_max)
    print("min_key is:", min_key, "min_file is:", min_file, "heart_min is:", heart_min)

    # generate max and min combined images
    # end_diastole max
    img = cv2.imread(os.path.join(testdata_folder, max_file))
    mask = cv2.imread(os.path.join(savedata_folder, max_file.split(".")[0]+".png"))
    max_image = cv2.addWeighted(img,1,mask,0.3,0)
    cv2.imwrite(os.path.join(media_dir, folder_name+"_end_diastole.jpg") , max_image)

    # end_systole min
    img = cv2.imread(os.path.join(testdata_folder, min_file))
    mask = cv2.imread(os.path.join(savedata_folder, min_file.split(".")[0]+".png"))
    min_image = cv2.addWeighted(img,1,mask,0.3,0)
    cv2.imwrite(os.path.join(media_dir, folder_name+"_end_systole.jpg") , min_image)

    return max_key, min_key, max_file, min_file, heart_max, heart_min, test_dict


# Calculate EF by using average area and L
def calculate_EF(max_key, min_key, heart_max, heart_min,
                    preprocess_input, gSess, gGraph, gModel,
                    testdata_folder, savedata_folder, test_dict, media_dir, folder_name,
                    average_count=5):
    # set parameters
    max_area_list = []
    min_area_list = []
    max_L_list = []
    min_L_list = []

    testdatalist = os.listdir(testdata_folder)
    total_images = len(testdatalist)

    # get the max list
    for i in range(average_count):
      if max_key+i-(average_count-1)/2 >= 0 and max_key+i-(average_count-1)/2 < total_images:
        image = io.imread(os.path.join(testdata_folder, test_dict[max_key+i-(average_count-1)/2]))
        # convert 1 channel to 3 channel if necessary
        if len(image.shape) == 2:
          image = np.stack((image,)*3, axis=-1)
        # print(test_dict[max_key+i-(average_count-1)/2])
        image = preprocess_input(image) # use the preprocessing
        image = np.expand_dims(image, axis = 0)
        # in django/flask predict the model with graph and session
        with gGraph.as_default():
            set_session(gSess)
            pr_mask = gModel.predict(image) # change the shape to (1 H, W)
        pr_mask = pr_mask.squeeze()
        pr_mask = np.argmax(pr_mask, axis = 2)
        # make sure the value is not too different from the original one
        if abs(heart_max - np.sum(pr_mask==1)) < heart_max*0.2:
          max_area_list.append(np.sum(pr_mask==1))
          _, _, r_max = get_enclosing_circle(pr_mask, media_dir, folder_name)
          max_L_list.append(2*r_max)

    # get the min list
    for i in range(average_count):
      if min_key+i-(average_count-1)/2 >= 0 and min_key+i-(average_count-1)/2 < total_images:
        image = io.imread(os.path.join(testdata_folder, test_dict[min_key+i-(average_count-1)/2]))
        # convert 1 channel to 3 channel if necessary
        if len(image.shape) == 2:
          image = np.stack((image,)*3, axis=-1)
        # print(test_dict[min_key+i-(average_count-1)/2])
        image = preprocess_input(image) # use the preprocessing
        image = np.expand_dims(image, axis = 0)
        # in django/flask predict the model with graph and session
        with gGraph.as_default():
            set_session(gSess)
            pr_mask = gModel.predict(image) # change the shape to (1 H, W)
        pr_mask = pr_mask.squeeze()
        pr_mask = np.argmax(pr_mask, axis = 2)
        # make sure the value is not too different from the original one
        if abs(heart_min - np.sum(pr_mask==1)) < heart_min*0.2:
          min_area_list.append(np.sum(pr_mask==1))
          _, _, r_min = get_enclosing_circle(pr_mask, media_dir, folder_name)
          min_L_list.append(2*r_min)

    # print results
    print("the max area is:", max_area_list)
    print("the min area is:", min_area_list)
    print("the L of max area is:", max_L_list)
    print("the L of min area is:", min_L_list)

    heart_max = sum(max_area_list) / len(max_area_list)
    heart_min = sum(min_area_list) / len(min_area_list)
    L_max = sum(max_L_list) / len(max_L_list)
    L_min = sum(min_L_list) / len(min_L_list)

    EF = round((heart_max**2/L_max - heart_min**2/L_min)/(heart_max**2/L_max),2)*100
    print("EF((EDV-ESV)/EDV):", EF)
    return EF


# genetate combined donut image
def make_donut(max_file, min_file, testdata_folder, media_dir, folder_name,
                preprocess_input, gSess, gGraph, gModel):
    # get the max image
    image = io.imread(os.path.join(testdata_folder, max_file))
    # convert 1 channel to 3 channel if necessary
    if len(image.shape) == 2:
      image = np.stack((image,)*3, axis=-1)

    image = preprocess_input(image) # use the preprocessing
    image = np.expand_dims(image, axis = 0)
    # in django/flask predict the model with graph and session
    with gGraph.as_default():
        set_session(gSess)
        pr_mask = gModel.predict(image) # change the shape to (1 H, W)
    # save mask for later combined image use
    pr_mask_max = pr_mask
    pr_mask_max = pr_mask_max.squeeze()
    pr_mask_max = np.argmax(pr_mask_max, axis = 2)

    # get the min image
    image = io.imread(os.path.join(testdata_folder, min_file))
    # convert 1 channel to 3 channel if necessary
    if len(image.shape) == 2:
      image = np.stack((image,)*3, axis=-1)
    image = preprocess_input(image) # use the preprocessing
    image = np.expand_dims(image, axis = 0)
    # in django/flask predict the model with graph and session
    with gGraph.as_default():
        set_session(gSess)
        pr_mask = gModel.predict(image) # change the shape to (1 H, W)
    # save mask for later combined image use
    pr_mask_min = pr_mask
    pr_mask_min = pr_mask_min.squeeze()
    pr_mask_min = np.argmax(pr_mask_min, axis = 2)

    # min image process
    pr_mask_min_t = np.where(pr_mask_min == 2, 0, pr_mask_min) # make unlabelled as background
    pr_mask_min_t = pr_mask_min_t * -1 # make the systole label as -1 for later calculation
    unique, counts = np.unique(pr_mask_min_t, return_counts=True)
    print(dict(zip(unique, counts)))
    # max image process
    pr_mask_max_t = np.where(pr_mask_max == 2, 0, pr_mask_max) # make unlabelled as background
    unique, counts = np.unique(pr_mask_max_t, return_counts=True)
    print(dict(zip(unique, counts)))
    # combine two images
    pr_mask_combined = pr_mask_min_t + pr_mask_max_t
    unique, counts = np.unique(pr_mask_combined, return_counts=True)
    print(dict(zip(unique, counts)))
    # save the donut image
    pr_mask_combined_draw = draw_combined_color(pr_mask_combined)
    io.imsave(os.path.join(media_dir, folder_name+"_donut.png"), pr_mask_combined_draw)


# utility function to preprocess an image and show the CAM
def classification(img_path, preprocess_input, gSess, gGraph, labels, gGap_weights,
                    gModel_class_cam, media_dir, folder_name):
    # load image
    img = cv2.imread(img_path)
    # preprocess the image before feeding it to the model
    img = cv2.resize(img, (512,512)) / 255.0
    # add a batch dimension because the model expects it
    image_value = np.expand_dims(img, axis=0)
    # get the features and prediction
    with gGraph.as_default():
        set_session(gSess)
        features, results = gModel_class_cam.predict(image_value)
        # generate the CAM
        # there is only one image in the batch so we index at `0`
        features_for_img = features[0]
        prediction = results[0]
        # there is only one unit in the output so we get the weights connected to it
        class_activation_weights = gGap_weights[:,0]
        # upsample to the image size
        class_activation_features = sp.ndimage.zoom(features_for_img, (512/64, 512/64, 1), order=2)
        # compute the intensity of each feature in the CAM
        cam_output  = np.dot(class_activation_features, class_activation_weights)
        # visualize the results
        print("the predict result is:", labels[np.argmax(results[0])])
        # save CAM image
        cam_output = Image.fromarray(cam_output).convert('RGB')
        cam_output.save(os.path.join(media_dir, folder_name+"_cam.png"))
        cam_output = cv2.imread(os.path.join(media_dir, folder_name+"_cam.png"))
        cam_output = cv2.applyColorMap(cam_output, cv2.COLORMAP_JET)
        image_value = np.squeeze(image_value).astype(np.uint8) * 255
        combined = cv2.addWeighted(cam_output,1,image_value,1,0)
        cv2.imwrite(os.path.join(media_dir, folder_name+"_cam.png"), combined)

    return labels[np.argmax(results[0])]
