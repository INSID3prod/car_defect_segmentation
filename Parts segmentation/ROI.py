import numpy as np
from numpy.matlib import empty
from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import random
import kornia as K
from itertools import combinations

from json_to_yolo import CLASS_ID_CLASS_NAME, CAR_BODY_PARTS

NUM_IMGS = 20
AREA_THRESH = 2000
DEVICE = torch.device('cuda:0')


test_imgs_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car damages dataset\File1\img')
color_test_imgs_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car damages dataset\File1\parts wrong colors')

SINGLE_INSTANCE_CLASSES = {
    0: 'Windshield',
    1: 'Back-windshield',
    16: 'Roof',
    12: 'Hood',
    13: 'Trunk',
    17: 'Grille',
    14: 'License-plate'
}

def filter_single_instance_classes(predict):
    boxes = predict[0].boxes
    masks = predict[0].masks

    if boxes is None or len(boxes) == 0:
        return boxes.data, masks.data

    keep = []

    for class_id in boxes.cls.unique():
        class_id = int(class_id.item())

        indices = torch.where(boxes.cls == class_id)[0]

        if class_id in SINGLE_INSTANCE_CLASSES:
            best_idx = indices[boxes.conf[indices].argmax()]
            keep.append(best_idx)
        else:
            keep.extend(indices)

    keep = torch.stack(keep)

    return boxes.data[keep], masks.data[keep]

def convert_to_Lab(rgb_tensor):
    return K.color.rgb_to_lab(rgb_tensor)

def get_ROI(predict, raw_img, size, show = True, color_mode: str = 'rgb'):
    raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)
    raw_img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).to(DEVICE)
    h,w = raw_img_tensor.shape[1:]

    bboxes, masks = filter_single_instance_classes(predict)
    #bboxes = predict[0].boxes.data
    #masks = predict[0].masks.data

    masks = F.interpolate(masks.unsqueeze(0),(h,w),mode='nearest').squeeze(0)

    ROI_dict = {}
    ROI_dict_Lab = {}

    car_mean_L = 0
    car_mean_a = 0
    car_mean_b = 0

    for part in range(bboxes.shape[0]):
        mask_area = masks[part].sum().item()
        # print(mask_area.item())
        if mask_area < AREA_THRESH:
            continue

        x1, y1, x2, y2 = bboxes[part, :4].round().int().tolist()
        class_id = int(bboxes[part,-1].item())

        if class_id not in CAR_BODY_PARTS.keys():
            continue

        cropped_mask = masks[part, y1:y2, x1:x2].unsqueeze(0)
        cropped_img = raw_img_tensor[:, y1:y2, x1:x2] / 255.0

        #cropped_mask = T.GaussianBlur((5,5),sigma=3)(cropped_mask)
        if size is not None:
            cropped_mask = F.interpolate(cropped_mask.unsqueeze(0).float(), (h, w), mode='bilinear', align_corners=False).squeeze(0)
            cropped_img = F.interpolate(cropped_img.unsqueeze(0).float(), (h, w), mode='bilinear', align_corners=False).squeeze(0)

        masked_img = (cropped_img * cropped_mask)

        #masked_img = F.pad(masked_img, (0, pad_w, 0, pad_h), value=127.0)
        ROI_dict[part] = masked_img.unsqueeze(0), class_id
        if color_mode == 'Lab':
            Lab_img = convert_to_Lab(cropped_img)

            mask = cropped_mask.squeeze(0) > 0.5

            L_mean = Lab_img[0][mask].mean().item()
            a_mean = Lab_img[1][mask].mean().item()
            b_mean = Lab_img[2][mask].mean().item()

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            ROI_dict_Lab[part] = {
                'Lab_image': Lab_img,
                'mask': cropped_mask,
                'Lab_metrics': {
                    'L_mean': L_mean,
                    'a_mean': a_mean,
                    'b_mean': b_mean
                },
                'class_id': class_id,
                'class_name': CLASS_ID_CLASS_NAME[class_id],
                'center': (cx, cy)
            }
            car_mean_L += L_mean
            car_mean_a += a_mean
            car_mean_b += b_mean

    car_mean_L /= bboxes.shape[0]
    car_mean_a /= bboxes.shape[0]
    car_mean_b /= bboxes.shape[0]
    car_mean = (car_mean_L, car_mean_a, car_mean_b)

    if show:
        for part in ROI_dict.keys():
            roi, class_id = ROI_dict[part]
            class_name = CLASS_ID_CLASS_NAME[class_id]
            img_2_show = ROI_dict[part][0].cpu().squeeze(0).permute(1, 2, 0).numpy()
            img_2_show = cv2.cvtColor(img_2_show, cv2.COLOR_RGB2BGR)
            cv2.imshow(f'ROI {class_name}', img_2_show)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    if color_mode == 'Lab':
        return ROI_dict, ROI_dict_Lab, car_mean
    else:
        return  ROI_dict

def car_parts_color_inspection(predict, raw_img):
    ROI, ROI_Lab, car_mean = get_ROI(predict, raw_img, None, True, 'Lab')

    all_parts = []

    for part in ROI_Lab:
        L = ROI_Lab[part]['Lab_metrics']['L_mean']
        a = ROI_Lab[part]['Lab_metrics']['a_mean']
        b = ROI_Lab[part]['Lab_metrics']['b_mean']
        delta_L_with_car_mean = (L - car_mean[0]) ** 2
        delta_ab_with_car_mean = (a - car_mean[1]) ** 2 + (b - car_mean[2]) ** 2
        
        part_name = ROI_Lab[part]['class_name']

        all_parts.append({
            'name': part_name,
            'center': ROI_Lab[part]['center'],
            'L': L,
            'a': a,
            'b': b,
            'delta_L': delta_L_with_car_mean,
            'delta_ab': delta_ab_with_car_mean
        })
        print(f'{part_name}:  delta_L = {delta_L_with_car_mean},  delta_ab = {delta_ab_with_car_mean}')

    distances = []

    for i, part in enumerate(all_parts):
        print(i, part['name'], part['center'])

    for part1, part2 in combinations(all_parts, 2):
        x1, y1 = part1['center']
        x2, y2 = part2['center']

        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        distances.append((distance, part1, part2))

        # print(f'distance {part1, part2} = {distance}')

    distances.sort()

    used_classes = set()
    min_distance_parts = []
    min_distance_metrics = []

    for distance, part1, part2 in distances:
        if part1['name'] in used_classes:
            continue

        min_distance_parts.append((part1, part2, distance))

        used_classes.add(part1['name'])
        used_classes.add(part2['name'])

        L1, L2 = part1['L'], part2['L']
        a1, a2 = part1['a'], part2['a']
        b1, b2 = part1['b'], part2['b']

        delta_ab = np.sqrt(
            (a1 - a2) ** 2 + (b1 - b2) ** 2
        )

        delta_E = np.sqrt(
            (L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2
        )

        min_distance_metrics.append({
            'parts': (part1, part2),
            'metrics': {
                'delta ab': delta_ab,
                'delta E': delta_E
            }
        })

        print(part1['name'], '<->', part2['name'], 'distance =', distance, f'  Color metrics: delta_ab = {delta_ab}')
    print('')
    return min_distance_metrics

def main():

    MODEL = YOLO(r'Z:\PYTHON\projects\Car damage detector\Parts segmentation\results\train-2\weights\best.pt')

    test_imgs = [_ for _ in test_imgs_path.iterdir()]
    color_test_imgs = [_ for _ in color_test_imgs_path.iterdir()]

    random.shuffle(test_imgs)

    for img in color_test_imgs:
        predict = MODEL.predict(img, verbose=False)
        raw_img = cv2.imread(img)
        min_distance_metrics = car_parts_color_inspection(predict, raw_img)

    for i in range(NUM_IMGS):
        predict = MODEL.predict(test_imgs[i], verbose=False)
        raw_img = cv2.imread(test_imgs[i])

        min_distance_metrics = car_parts_color_inspection(predict, raw_img)


if __name__ == '__main__':
    main()