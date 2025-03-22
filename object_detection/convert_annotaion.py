import json
import os


classes = {
    'vehicle': 0,
    'pedestrian': 1,
    'traffic-sign': 2,
    'traffic-light-green': 3,
    'traffic-light-red': 4,
    'traffic-light-etc': 5,
    'crosswalk': 6,
    'two-wheeled-vehicle': 7,
    'bike': 8
}

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
            class_id = classes[category]

            annotation_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        base_name = os.path.splitext(image_name)[0]
        output_file = os.path.join(output_folder, base_name + '.txt')
        with open(output_file, 'w') as f:
            for line in annotation_lines:
                f.write(line + "\n")


if __name__ == '__main__':
    json_dir = os.path.join('.', 'dataset', 'validation', 'annotation')
    output_folder = os.path.join('.', 'labels')

    convert_annotations(json_dir, output_folder)
