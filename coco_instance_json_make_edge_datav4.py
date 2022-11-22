import numpy as np
import cv2
import os 
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from matplotlib import pyplot as plt
from detectron2.data.detection_utils import annotations_to_instances
def ifnotexistmakedir(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"
if __name__ == '__main__':
    dataset_list = ['coco','lvis','isod','TPVOCIS']
    dataset_list_select = dataset_list[3]
    if dataset_list_select=='coco':
        dataset_dir = './datasets/coco/'
        json_file = dataset_dir+'annotations/instances_train2017.json'
        imgs_dir = dataset_dir+'train2017/'
        contour_output_dir = dataset_dir+'contour/'
        GT_output_dir = dataset_dir+'GT/'
        is_output_GT =False
    if dataset_list_select=='lvis':
        dataset_dir = './datasets/lvis/'
        json_file = dataset_dir+'lvis_v1_train.json'
        coco_dataset_dir = './datasets/coco/'
        imgs_dir = coco_dataset_dir+'train2017/'
        contour_output_dir = dataset_dir+'contour/'
        GT_output_dir = dataset_dir+'GT/'
        is_output_GT =False

    if dataset_list_select=='isod':
        json_file = './detectron2_code/dataset/isod/train.json'
        dataset_dir = './detectron2_code/dataset/isod/'
        imgs_dir = dataset_dir+'imgs/'
        contour_output_dir = dataset_dir+'contour/'
        GT_output_dir = dataset_dir+'GT/'
        is_output_GT =True
    if dataset_list_select=='TPVOCIS':
        json_file = './detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/trainv2.json'
        dataset_dir = './detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/'
        imgs_dir = dataset_dir+'train_images/'
        contour_output_dir = dataset_dir+'contour/'
        GT_output_dir = dataset_dir+'GT/'
        is_output_GT =False
    coco = COCO(json_file)
    catIds = coco.getCatIds() # 
    all_categories = coco.loadCats(catIds)
    print(len(catIds))
    imgIds = coco.getImgIds() # 圖片id，許多值
    categories_list = [category['name'] for category in all_categories]             
    print(categories_list)

    print(len(imgIds))
    if is_output_GT:
        ifnotexistmakedir([contour_output_dir,GT_output_dir])
    else:
        ifnotexistmakedir([contour_output_dir])
        
    """
    with open(json_file) as anno_:
        all_annotations = json.load(anno_)

    annotations = all_annotations['annotations']
    """
    results = []
    plt.axis('off')   
    for dataset_name in ['train']:
        has_contour = 0
        lose = 0
        for i,imgId in enumerate(imgIds):
            print('Now Image',i)
            img = coco.loadImgs(imgId)
            #print(img)
            #print('{}{}'.format(imgs_dir,img[0]['file_name']))
            if type(img) is list:
                img = img[0]
            else:                 
                print('img is not list!!!!!!!!!!!')                 
                exit()
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)     
            anns = coco.loadAnns(annIds)

            width = img['width']
            height = img['height']
            
            instance_mask = np.zeros((height, width))
            contour_image = np.zeros((height, width))
            for j,single in enumerate(anns):
                mask_single = coco.annToMask(single)
                if is_output_GT:
                    className = getClassName(anns[j]['category_id'], all_categories)
                    pixel_value = categories_list.index(className)+1
                    instance_mask += np.maximum(mask_single*pixel_value, mask_single)
                    

                for row in range(height):
                    for col in range(width):
                        if (mask_single[row][col] > 0):   
                            mask_single[row][col] = 255

                imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
                imgs[:, :, 0] = mask_single[:, :]
                imgs[:, :, 1] = mask_single[:, :]
                imgs[:, :, 2] = mask_single[:, :]
                imgs = imgs.astype('uint8')
                plt.axis('off')
                
                
                
                image = imgs[...,::-1]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cnts, hei = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contour_image_original = contour_image.copy()
                cv2.drawContours(contour_image, cnts, -1, (255, 255, 255), 1)
                cv2.imwrite(contour_output_dir+img['file_name'][:-4]+'.png', contour_image)

                if np.array_equal(contour_image_original,contour_image):
                    lose+=1
                    print(img['file_name'][:-4]+'.png')
                else:
                    has_contour+=1
                
            if is_output_GT:
                plt.imshow(instance_mask)
                plt.savefig(GT_output_dir+img['file_name'][:-4]+'.png', bbox_inches='tight', pad_inches=0)
    print(' has_contour = ',has_contour)
    print(' lose = ',lose)
            
            


