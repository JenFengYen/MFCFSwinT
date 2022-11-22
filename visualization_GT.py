import os
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor


if __name__ == '__main__':
    register_coco_instances(name='cod10k_train',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Train/CAM_Instance_Train_v2.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Train/Image/')
    register_coco_instances(name='cod10k_test',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Test/CAM_Instance_Test_v2.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Test/Image/')
    #
    
    
    #cfg.MODEL.BACKBONE.NAME = 'build_tv_wide_resnet50_2_fpn_backbone'
    #cfg.LKDBlock = False
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    #cfg.DATASETS.TEST = ("cod10k_train", )
    #predictor = DefaultPredictor(cfg)
    from detectron2.data import DatasetCatalog
    #from detectron2.utils.visualizer import Visualizer
    from visualizer_label_frontsize_fix import Visualizer
    #from visualizer_label_frontsize_fix_no_box_no_class import Visualizer
    #from visualizer_label_frontsize_fix_no_instance import Visualizer
    from detectron2.utils.visualizer import ColorMode
    dataset_name = "coco_2017_val"
    dataset_dicts = DatasetCatalog.get(dataset_name)
    i=0
    if dataset_name == "cod10k_test":
        image_choose = ['COD10K-CAM-1-Aquatic-7-Flounder-266.jpg','COD10K-CAM-1-Aquatic-9-GhostPipefish-399.jpg','COD10K-CAM-3-Flying-50-Bat-2960.jpg','COD10K-CAM-3-Flying-53-Bird-3024.jpg']
        OUTPUT_DIR = "./detectron2_code/result/cod10k/test_GT/"
    if dataset_name == "coco_2017_val":
        image_choose = ['000000004495.jpg','000000005992.jpg','000000011760.jpg','000000026204.jpg' ]
        OUTPUT_DIR = "./detectron2_code/result/coco_remove_iscrowd/val_GT/"
    if dataset_name == "lvis_v1_val":
        image_choose = ['000000003264.jpg','000000007139.jpg','000000009673.jpg','000000023038.jpg']
        OUTPUT_DIR = "./detectron2_code/result/LVISv1_remove_iscrowd/val_GT/"
    os.makedirs(OUTPUT_DIR,exist_ok=True) 
    for d in dataset_dicts:  
        for img_choose in image_choose:
            if os.path.basename(d["file_name"]) == img_choose:
                #outputs = predictor(im)
                #print(outputs["instances"])
                img = cv2.imread(d["file_name"])
                print(os.path.basename(d["file_name"]).replace('jpg','png'))
                visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0)     
                out = visualizer.draw_dataset_dict(d)
                #mask = outputs["instances"].get("pred_masks")
                #out_path = os.path.join(cfg.OUTPUT_DIR,'mask_result/') + os.path.basename(d["file_name"]).replace('jpg','png')

                cv2.imwrite(os.path.join(OUTPUT_DIR,os.path.basename(d["file_name"]).replace('jpg','png')),out.get_image()[:, :, ::-1])

