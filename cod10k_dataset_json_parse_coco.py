import json
import shutil
import re
if __name__ == '__main__':
    Sub_Class_Dictionary = {'1':'batFish','2':'clownFish','3':'crab','4':'crocodile','5':'crocodileFish','6':'fish','7':'flounder','8':'frogFish','9':'ghostPipefish','10':'leafySeaDragon','11':'octopus','12':'pagurian','13':'pipefish','14':'scorpionFish','15':'seaHorse','16':'shrimp','17':'slug','18':'starFish','19':'stingaree','20':'turtle','21':'ant','22':'bug','23':'cat','24':'caterpillar','25':'centipede','26':'chameleon','27':'cheetah','28':'deer','29':'dog','30':'duck','31':'gecko','32':'giraffe','33':'grouse','34':'human','35':'kangaroo','36':'leopard','37':'lion','38':'lizard','39':'monkey','40':'rabbit','41':'reccoon','42':'sciuridae','43':'sheep','44':'snake','45':'spider','46':'stickInsect','47':'tiger','48':'wolf','49':'worm','50':'bat','51':'bee','52':'beetle','53':'bird','54':'bittern','55':'butterfly','56':'cicada','57':'dragonfly','58':'frogmouth','59':'grasshopper','60':'heron','61':'katydid','62':'mantis','63':'mockingbird','64':'moth','65':'owl','66':'owlfly','67':'frog','68':'toad','69':'other'}
    json_root='./detectron2_code/COD10K-v3/Train/CAM_Instance_Train.json'
    json_root_new ='./detectron2_code/COD10K-v3/Train/CAM_Instance_Train_v2.json'
    #json_root='./detectron2_code/COD10K-v3/Test/CAM_Instance_Test.json'
    #json_root_new = './detectron2_code/COD10K-v3/Test/CAM_Instance_Test_v2.json'
    json_result=[]
    class_count=69
    with open(json_root) as f:  
        annotation = json.load(f)
        categories=[]
        for i in range(class_count):
            categories.append({"supercategory":Sub_Class_Dictionary[str(i+1)],"id": i+1,"name": Sub_Class_Dictionary[str(i+1)]})
        annotation["categories"]=categories
        for i in range(len(annotation["annotations"])):
            image_id = annotation["annotations"][i]["image_id"]
            file_name = annotation["images"][image_id-1]["file_name"]        
            category_name = file_name.split('-')[-2]
            category_id = -1
            for j in range(class_count):
                if category_name.lower() == Sub_Class_Dictionary[str(j+1)].lower():
                    category_id = j+1
                    break
            annotation["annotations"][i]["category_id"] = category_id

        json_result = annotation
        f.close()
    shutil.copyfile(json_root, json_root_new)
    with open(json_root_new,'w') as w: 
        json.dump(json_result,w, indent=4)
        w.close()