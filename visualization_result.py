import os
import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
import backbone.swin_transformer_pkd
import backbone.meta_arch.rcnn
if __name__ == '__main__':
    register_coco_instances(name='cod10k_train',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Train/CAM_Instance_Train_v2.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Train/Image/')
    register_coco_instances(name='cod10k_test',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Test/CAM_Instance_Test_v2.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Test/Image/')
    model_dir = "./detectron2_code/result/LVISv1_remove_iscrowd/swint_T_MFFv3_PKD_Edegev3_CascadeROIHeads_pretrain_imagenet1K_freeze_3_lr_0.0001_step_50_60_batch_5_2_lr_sched2x/"
    #model_dir = "./detectron2_code/result/coco_remove_iscrowd/swint_T_MFFv3_PKD_Edegv3_CascadeROIHeads_pretrain_imagenet1K_freeze_3_lr_0.0001_batch_5_lr_sched3x/"
    #model_dir = "./detectron2_code/result/cod10k/swint_T_MFFv3_PKD_Edegev3_CascadeROIHeads_pretrain_imagenet1K_freeze_3_lr_0.0001_step_50_60_batch_5_EPOCH_200/"
    OUTPUT_DIR = model_dir+"result_SCORE_THRESH_TEST_0.5/"
    
    os.makedirs(OUTPUT_DIR,exist_ok=True) 

    import pickle 
    filename = os.path.join(model_dir, 'config.pkl') 
    with open(filename, 'rb') as f:      
        cfg = pickle.load(f)
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_best.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    dataset_name = "lvis_v1_val"
    cfg.DATASETS.TEST = (dataset_name, )
    predictor = DefaultPredictor(cfg)
    from detectron2.data import DatasetCatalog
    from visualizer_label_frontsize_fix import Visualizer
    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = DatasetCatalog.get(dataset_name)
    i=0
    if dataset_name == "cod10k_test":
        image_choose = ['COD10K-CAM-1-Aquatic-7-Flounder-266.jpg','COD10K-CAM-1-Aquatic-9-GhostPipefish-399.jpg','COD10K-CAM-3-Flying-50-Bat-2960.jpg','COD10K-CAM-3-Flying-53-Bird-3024.jpg']
    if dataset_name == "coco_2017_val":
        image_choose = ['000000004495.jpg','000000005992.jpg','000000011760.jpg','000000026204.jpg' ]
    if dataset_name == "lvis_v1_val":
        image_choose = ['000000003264.jpg','000000007139.jpg','000000009673.jpg','000000023038.jpg']
    for d in dataset_dicts:  
        for img_choose in image_choose:     
            if os.path.basename(d["file_name"]) == img_choose:
                img = cv2.imread(d["file_name"])
                outputs = predictor(img)
                print(os.path.basename(d["file_name"]).replace('jpg','png'))
                visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0)     
                out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
                #mask = outputs["instances"].get("pred_masks")
                #out_path = os.path.join(cfg.OUTPUT_DIR,'mask_result/') + os.path.basename(d["file_name"]).replace('jpg','png')

                cv2.imwrite(os.path.join(OUTPUT_DIR,os.path.basename(d["file_name"]).replace('jpg','png')),out.get_image()[:, :, ::-1])

