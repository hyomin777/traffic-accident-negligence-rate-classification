import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from config import OBJECT_CLASSES


def convert_annotations(dir_path, output_folder):
    file_list = os.listdir(dir_path)

    for file in file_list:
        file_path = os.path.join(dir_path, file)
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

            annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        base_name = os.path.splitext(image_name)[0]
        output_file = os.path.join(output_folder, base_name + '.txt')
        with open(output_file, 'w') as f:
            for line in annotation_lines:
                f.write(line + "\n")


if __name__ == '__main__':
    json_dir = os.path.join('.', 'datasets', 'train', 'annotation')
    output_folder = os.path.join('.', 'datasets', 'train', 'data')
    convert_annotations(json_dir, output_folder)

    json_dir = os.path.join('.', 'datasets', 'validation', 'annotation')
    output_folder = os.path.join('.', 'datasets', 'validation', 'data')
    convert_annotations(json_dir, output_folder)
