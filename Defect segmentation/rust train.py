from pathlib import Path
import time
import torch
import os

from ultralytics import YOLO

data_yaml = r'Z:\PYTHON\projects\Car damage detector\dataset\rust.v1i.yolo26\data.yaml'
save_models_path = r'Z:\PYTHON\projects\Car damage detector\Defect segmentation\models'
results_path = r'Z:\PYTHON\projects\Car damage detector\Defect segmentation\rust results 1'

MODEL = YOLO(r'Z:\PYTHON\projects\Car damage detector\Defect segmentation\models\rust_seg.pt')

EPOCHS = 20
size = 640

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
print(f'using {DEVICE}, version {torch.version.cuda}')


def main():
    print("-------------TRAIN---------------")

    start_time = time.time()


    MODEL.train(data=data_yaml,
                epochs=EPOCHS,
                imgsz=size,
                batch=1,
                save=True,
                device=DEVICE,
                project=results_path,
                verbose=False
                )

    end_time = time.time()

    print(f'training was {end_time - start_time} long')

    print("-------------VAL---------------")

    start_time = time.time()

    MODEL.val(
        data=data_yaml,
        split='val',
        imgsz=size,
        batch=1,
        device=DEVICE,
        project=results_path,
        name='val',
        verbose=False
    )

    end_time = time.time()

    print(f'validation was {end_time - start_time} long')

    print("-------------TEST---------------")

    start_time = time.time()

    MODEL.val(
        data=data_yaml,
        split='test',
        imgsz=size,
        batch=1,
        device=DEVICE,
        project=results_path,
        name='test',
        verbose=False
    )

    end_time = time.time()

    print(f'test was {end_time - start_time} long')

    MODEL.save(os.path.join(save_models_path, 'rust_seg_1.pt'))

    print(f'model saved to {save_models_path}')


if __name__ == '__main__':
    main()