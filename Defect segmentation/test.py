from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F

from json_to_yolo import CLASS_TO_ID

MODEL = YOLO(r'Z:\PYTHON\projects\Car damage detector\Parts segmentation\results\train-2\weights\best.pt')
DEVICE = torch.device('cuda:0')

test_imgs_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car damages dataset\File1\img')

test_imgs = [_ for _ in test_imgs_path.iterdir()]

NUM_IMGS = 20
AREA_THRESH = 1000

def find_keys_by_value(d, value):
    return [key for key, val in d.items() if val == value]

def get_ROI(predict, raw_img, size: tuple, show = True):
    raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)
    raw_img_torch = torch.from_numpy(raw_img).permute(2,0,1).to(DEVICE)
    h,w = raw_img_torch.shape[-2:]

    bboxes = predict[0].boxes.data
    masks = predict[0].masks.data

    masks = F.interpolate(masks.unsqueeze(0),(h,w),mode='nearest').squeeze(0)

    ROI_dict = {}
    ROI_dict_resized = {}

    for ch in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[ch, :4].round().int().tolist()
        class_id = bboxes[ch,-1].int()

        mask_area = masks[ch].sum().item()
        #print(mask_area.item())
        if mask_area < AREA_THRESH:
            continue

        masked_img = masks[ch, y1:y2, x1:x2] * raw_img_torch[:, y1:y2, x1:x2]

        ROI_dict[ch] = masked_img, class_id
        ROI_dict_resized[ch] = F.interpolate(masked_img.unsqueeze(0), size, mode='nearest').squeeze(0), class_id
    if show:
        ch, (roi, class_id) = next(iter(ROI_dict_resized.items()))
        class_name = find_keys_by_value(CLASS_TO_ID, class_id)
        img_2_show = ROI_dict_resized[0][0].cpu().permute(1, 2, 0).numpy()
        img_2_show = cv2.cvtColor(img_2_show,cv2.COLOR_RGB2BGR)
        cv2.imshow(f'ROI of {class_name}', img_2_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if size is not None:
        return ROI_dict_resized
    else:
        return ROI_dict

for i in range(NUM_IMGS):
    predict = MODEL.predict(test_imgs[i])
    raw_img = cv2.imread(test_imgs[i])
    #predict[0].show()
    ROI = get_ROI(predict,raw_img,(640,640),True)