import json
from pathlib import Path

ann_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car damages dataset\File1\ann')
yolo_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car damages dataset\File1\ann_yolo')
img_path = Path(r'Z:\PYTHON\projects\Car damage detector\dataset\Car damages dataset\File1\img')

CLASS_NAMES = [
    'Dent',
    'Cracked',
    'Scratch',
    'Flaking',
    'Broken part',
    'Paint chip',
    'Missing part',
    'Corrosion'
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


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