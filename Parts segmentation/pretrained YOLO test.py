from ROI import get_ROI, test_imgs_path

from ultralytics import YOLO

import cv2
import random

parts_MODEL = YOLO(r'Z:\PYTHON\projects\Car damage detector\Parts segmentation\results\train-2\weights\best.pt')
defect_MODEL = YOLO(r'Z:\PYTHON\projects\Car damage detector\Defect segmentation\models\damage_segmentation.pt')

test_imgs = [_ for _ in test_imgs_path.iterdir()]

#test_img = test_imgs[random.randint(0,len(test_imgs))]
test_img = test_imgs[117]
test_img = cv2.imread(test_img)

parts_pred = parts_MODEL.predict(test_img)

ROI = get_ROI(parts_pred,test_img,(896,896),False)

defects_pred = defect_MODEL(test_img,verbose=True)
defects_pred[0].show()


for instance in ROI.keys():
    roi_img, class_id = ROI[instance]

    defects_pred = defect_MODEL(roi_img,verbose=False)
    defects_pred[0].show()
