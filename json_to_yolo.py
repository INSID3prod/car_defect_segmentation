import json
from pathlib import Path

ann_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car parts dataset\File1\ann')
yolo_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car parts dataset\File1\ann_yolo')
img_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car parts dataset\File1\img')

CLASS_NAMES = [
    'Windshield',
    'Back-windshield',
    'Front-window',
    'Back-window',
    'Front-door',
    'Back-door',
    'Front-wheel',
    'Back-wheel',
    'Front-bumper',
    'Back-bumper',
    'Headlight',
    'Tail-light',
    'Hood',
    'Trunk',
    'License-plate',
    'Mirror',
    'Roof',
    'Grille',
    'Rocker-panel',
    'Quarter-panel',
    'Fender'
]

CLASS_NAME_CLASS_ID = {
    'Windshield': 0,
    'Back-windshield': 1,
    'Front-window': 2,
    'Back-window': 3,
    'Front-door': 4,
    'Back-door': 5,
    'Front-wheel': 6,
    'Back-wheel': 7,
    'Front-bumper': 8,
    'Back-bumper': 9,
    'Headlight': 10,
    'Tail-light': 11,
    'Hood': 12,
    'Trunk': 13,
    'License-plate': 14,
    'Mirror': 15,
    'Roof': 16,
    'Grille': 17,
    'Rocker-panel': 18,
    'Quarter-panel': 19,
    'Fender': 20
}

CLASS_ID_CLASS_NAME = {
    0: 'Windshield',
    1: 'Back-windshield',
    2: 'Front-window',
    3: 'Back-window',
    4: 'Front-door',
    5: 'Back-door',
    6: 'Front-wheel',
    7: 'Back-wheel',
    8: 'Front-bumper',
    9: 'Back-bumper',
    10: 'Headlight',
    11: 'Tail-light',
    12: 'Hood',
    13: 'Trunk',
    14: 'License-plate',
    15: 'Mirror',
    16: 'Roof',
    17: 'Grille',
    18: 'Rocker-panel',
    19: 'Quarter-panel',
    20: 'Fender'
}

CAR_BODY_PARTS = {
    4 :'Front-door',
    5: 'Back-door',
    8: 'Front-bumper',
    9: 'Back-bumper',
    12: 'Hood',
    13: 'Trunk',
    16: 'Roof',
    18: 'Rocker-panel',
    19: 'Quarter-panel',
    20: 'Fender'
}



print(CLASS_NAME_CLASS_ID)

"""
def json_to_yolo(json_path, txt_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = data["size"]["width"]
    height = data["size"]["height"]

    print(json_path.name)
    print("objects:", len(data["objects"]))

    lines = []

    for obj in data["objects"]:
        class_name = obj["classTitle"]

        print("  class:", repr(class_name))

        if class_name not in CLASS_TO_ID:
            print("  !!! UNKNOWN CLASS")
            continue

        points = obj["points"]["exterior"]

        if len(points) < 3:
            print("  !!! TOO FEW POINTS")
            continue

        class_id = CLASS_TO_ID[class_name]

        coords = []

        for x, y in points:
            x_norm = x / width
            y_norm = y / height

            coords.extend([x_norm, y_norm])

        line = " ".join(
            [str(class_id)] +
            [f"{v:.6f}" for v in coords]
        )

        lines.append(line)

    print("lines:", len(lines))

    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

json_files = list(ann_path.glob("*.json"))

for json_path in json_files:
    image_name = json_path.name.removesuffix(".json")
    image_path = img_path / image_name

    if not image_path.exists():
        print(f"Нет картинки: {image_name}")
        continue

    txt_path = yolo_path / f"{image_path.stem}.txt"
    json_to_yolo(json_path, txt_path)

print(f"Обработано: {len(json_files)}")
"""