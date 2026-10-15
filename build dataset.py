from pathlib import Path
import shutil
import random

root = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car damages dataset')

yolo_path = root / 'File1' / 'ann_yolo'
img_path = root / 'File1' / 'img'

train_path = root / 'train'
val_path = root / 'val'
test_path = root / 'test'


pairs = []

for img in img_path.iterdir():
    txt = yolo_path / f'{img.stem}.txt'

    if txt.exists():
        pairs.append((img, txt))
    else:
        print(f'Нет разметки для: {img.name}')


random.seed(42)
random.shuffle(pairs)

# Split 80 / 10 / 10
n = len(pairs)

train_pairs = pairs[:int(0.8 * n)]
val_pairs = pairs[int(0.8 * n):int(0.9 * n)]
test_pairs = pairs[int(0.9 * n):]


def build_dataset(parent, pairs):
    path_imgs = parent / 'images'
    path_txt = parent / 'labels'

    path_imgs.mkdir(parents=True, exist_ok=True)
    path_txt.mkdir(parents=True, exist_ok=True)

    for img, txt in pairs:
        shutil.copy2(img, path_imgs / img.name)
        shutil.copy2(txt, path_txt / txt.name)


build_dataset(train_path, train_pairs)
build_dataset(val_path, val_pairs)
build_dataset(test_path, test_pairs)


print(f'Total: {n}')
print(f'Train: {len(train_pairs)}')
print(f'Val:   {len(val_pairs)}')
print(f'Test:  {len(test_pairs)}')