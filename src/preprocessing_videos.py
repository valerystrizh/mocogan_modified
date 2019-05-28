import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings

def extract_frames(file_addr,output, sheight=64,top_crop=0,left_crop=0,right_crop=-1,bottom_crop=-1):
    vid = cv2.VideoCapture(file_addr)
    success, frame = vid.read()
    frame = frame[top_crop:bottom_crop,left_crop:right_crop]
    if success:
        count = 1
        while True:
            success,tmp = vid.read()
            if not success:
                break
            tmp = tmp[top_crop:bottom_crop,left_crop:right_crop]
            frame = np.hstack((frame,tmp))
            
            count += 1
        print('Number of frame(s): {} done!'.format(count))
        width = frame.shape[1]
        height = frame.shape[0]
        scale_factor = sheight/height
        frame = cv2.resize(frame,(int(width*scale_factor),int(height*scale_factor)))
        write_stat = cv2.imwrite(output, frame)
        assert write_stat, 'Got problem in writing image.'
    else:
        warnings.warn('The file \"{}\" is corrupted. Couldn\'t open it.'.format(file_addr))
        
        
def preprocessing_videos(dataset_location = './dataset',output='./processed_dataset',vtype='.avi', sheight = 64,
    top_crop=0,left_crop=0,right_crop=-1,bottom_crop=-1):
    filt = list(filter(lambda x: os.path.isdir(os.path.join(dataset_location,x)), os.listdir(dataset_location)))
    assert len(filt)>0,\
    'You have to put your videos inside of their category subfolders and proceed pre-processing.'
    if not os.path.exists(output):
        os.mkdir(output)
    for i in filt:
        cur_path = os.path.join(dataset_location,i)
        cur_path_out = os.path.join(output,i)
        mov_filt = list(filter(lambda x: os.path.isfile(os.path.join(cur_path,x)) and x.endswith(vtype),os.listdir(cur_path)))
        print('{} \t count: {} \t {} file(s)'.format(i,len(mov_filt),vtype))
        if not os.path.exists(cur_path_out) and len(mov_filt)>0:
            os.mkdir(cur_path_out)
        for j in mov_filt:
            outfile = os.path.join(cur_path_out,j.replace(vtype,'.jpg'))
            inpfile = os.path.join(cur_path,j)
            print('({})'.format(inpfile),end=':   ')
            extract_frames(file_addr = inpfile, output = outfile, sheight=sheight, top_crop = top_crop, left_crop = left_crop, right_crop = right_crop, bottom_crop = bottom_crop)
        print('\n')