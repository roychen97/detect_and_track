# gen_random_peoplo
import cv2 as cv
import json
import os
import sys
import numpy as np
import random
from tqdm import tqdm

label_path = '/home/scchiu/Data/VOCdevkit/VOC2007_annotation_person' 
label_json_list = os.listdir(label_path)
outout_img_path =  '/home/scchiu/Data/VOCdevkit/VOC_merged' 
outout_json_path =  '/home/scchiu/Data/VOCdevkit/VOC_annotation_person_merged' 

print("sample num = {}".format(len(label_json_list)))

# x = [i for i in range(len(label_json_list))]        
# random.shuffle(x)
# label_json_list = [label_json_list[i] for i in x]






def gen_crop_data():
    global label_json_list
    p1 = random.randint(0,len(label_json_list)-1)
    
    
    # load all person bboxes
    with open(os.path.join(label_path, label_json_list[p1])) as json_file:
        data = json.load(json_file)
    img_path=data["img_name"]    
    img = cv.imread(img_path)
    persons = []

    objs = data["object"]
    min_y = img.shape[0]
    for obj in objs:
        bbox = [int(obj["x1"]), int(obj["y1"]), int(obj["x2"]), int(obj["y2"])]
        min_y = min(min_y, bbox[1])
        persons.append(bbox)
        # cv.rectangle(img, (int(obj["x1"]), int(obj["y1"])), (int(obj["x2"]), int(obj["y2"])), (255, 155, 255), 2)    
    # cv.imshow("image {}".format(p1), img)


    select_person = random.randint(0,len(persons)-1)
    bbox = persons[select_person]
    y_bottom = (bbox[1] + bbox[3])//2
    h = (bbox[3] - bbox[1])//3
    y_up = max(min_y - h, 0)


    if not y_bottom > y_up:
        return img, []

    new_persons = []
    for bbox in persons:
        x1 = bbox[0]
        x2 = bbox[2]
        y1 = min(bbox[1], y_bottom)
        y2 = min(bbox[3], y_bottom)
        y1 = max(y1-y_up, 0)
        y2 = max(y2-y_up, 0)
        if y2 > y1:
            new_persons.append([x1, y1, x2, y2])

    crop_img = img[y_up:y_bottom,:].copy()
    # for bbox in new_persons:
    #     cv.rectangle(crop_img, ( bbox[0],  bbox[1]), ( bbox[2],  bbox[3]), (0, 0, 255), 2)
    # cv.imshow("crop person {}".format(p1), crop_img)

    return crop_img, new_persons





for img_idx in tqdm(range(20000,30000)):
    filename = '{:05d}'.format(img_idx)

    crop_imgs = []
    person_boxes = []
    width = 1000000

    while len(crop_imgs) < 3:
        crop_img, new_persons = gen_crop_data()
        if len(new_persons)> 0:
            crop_imgs.append(crop_img)
            person_boxes.append(new_persons)
            width = min(width, crop_img.shape[1])
        
    resize_imgs = []
    merged_boxes = []
    add_y = 0


    object_list = []
    for img, bboxes in zip(crop_imgs, person_boxes):        
        resize_imgs.append(cv.resize(img, (width, width * img.shape[0] // img.shape[1])))
        ratio = width / img.shape[1]
        for bbox in bboxes:
            box = [int(bbox[0] * ratio), int(bbox[1] * ratio)+ add_y,
            int(bbox[2] * ratio), int(bbox[3] * ratio) + add_y]
            if box[3] - box[1] < 10 or box[2] - box[0] < 10:
                continue
            if bbox[3] - bbox[1] < 0.03 or bbox[2] - bbox[0] < 0.03:
                continue                
            merged_boxes.append(box)
            object_ = {
                        "class":str("person"),
                        "x1":float(merged_boxes[-1][0]),
                        "y1":float(merged_boxes[-1][1]),
                        "x2":float(merged_boxes[-1][2]),
                        "y2":float(merged_boxes[-1][3])
                        }
            object_list.append(object_)  
        add_y += resize_imgs[-1].shape[0]                    
        




    merged_img = np.concatenate(resize_imgs, axis = 0)
    tmp = cv.resize(merged_img, (200, 200), interpolation=cv.INTER_NEAREST)
    merged_img = cv.resize(tmp, (merged_img.shape[1], merged_img.shape[0]))


    full_path = os.path.join(outout_img_path, filename + '.jpg')
    cv.imwrite(full_path, merged_img)
    json_str_from_txt = {
                    "img_name":full_path,
                    "height": merged_img.shape[0],
                    "width": merged_img.shape[1],
                    "depth": merged_img.shape[2],
                    "object": object_list
                }

    json_str = json.dumps(json_str_from_txt, indent=2) 
    with open(os.path.join(outout_json_path,
        'Pascal_VOC_' + filename + '.json'), 'w') as w:
        w.write(json_str)



    # for bbox in merged_boxes:
    #     cv.rectangle(merged_img, ( bbox[0],  bbox[1]), ( bbox[2],  bbox[3]), (0, 0, 255), 2)
    #     print('box = {}'.format(bbox))
    # cv.imshow('merged_img', merged_img)  
    # cv.waitKey()  



test_json_list = os.listdir(outout_json_path)
for i, json_file in enumerate(test_json_list):
    if i > 10:
        break
    with open(os.path.join(outout_json_path, json_file)) as json_file:
        data = json.load(json_file)
    img_path=data["img_name"]    
    img = cv.imread(img_path)
    objs = data["object"]

    for obj in objs:
        cv.rectangle(img, (int(obj["x1"]), int(obj["y1"])), (int(obj["x2"]), int(obj["y2"])), (155, 55, 255), 1)    
    cv.imshow("test json image {}".format(i), img)
cv.waitKey()    

