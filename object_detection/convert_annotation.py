import sys
import os
import json
from config import OBJECT_CLASSES


def convert_annotations(dir_path, output_root):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    data = json.load(f)

                image_name = data['file_name']
                annotation_lines = []

                width = data['width']
                height = data['height']
                for obj in data['objects']:
                    x, y, w, h = obj['bbox']
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height

                    category = obj['category']
                    class_id = OBJECT_CLASSES[category]

                    annotation_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    )

                relative_path = os.path.relpath(root, dir_path)
                output_folder = os.path.join(output_root, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                base_name = os.path.splitext(image_name)[0]
                output_file = os.path.join(output_folder, base_name + '.txt')

                with open(output_file, 'w') as f:
                    for line in annotation_lines:
                        f.write(line + "\n")


if __name__ == '__main__':
    json_dir = os.path.join('.', 'object_detection', 'datasets', 'train', 'annotation')
    output_root = os.path.join('.', 'object_detection', 'datasets', 'train', 'data')

    convert_annotations(json_dir, output_root)
