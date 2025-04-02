import os
import json
from collections import defaultdict

def get_unique_class_counts(annotation_dir):
    accident_types = defaultdict(int)
    accident_places = defaultdict(int)
    accident_place_features = defaultdict(int)
    vehicle_a_progress = defaultdict(int)
    vehicle_b_progress = defaultdict(int)
    accident_negligence = defaultdict(int)

    for filename in os.listdir(annotation_dir):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(annotation_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        video_data = data.get("video", {})
        if "traffic_accident_type" in video_data:
            accident_types[video_data["traffic_accident_type"]] += 1
        elif "accident_type" in video_data:
            accident_types[video_data["accident_type"]] += 1
        else:
            accident_types['None'] += 1
        if "accident_place" in video_data:
            accident_places[video_data["accident_place"]] += 1
        else:
            accident_places['None'] += 1
        if "accident_place_feature" in video_data:
            accident_place_features[video_data["accident_place_feature"]] += 1
        else:
            accident_place_features['None'] += 1
        if "vehicle_a_progress_info" in video_data:
            vehicle_a_progress[video_data["vehicle_a_progress_info"]] += 1
        else:
            vehicle_a_progress['None'] += 1
        if "vehicle_b_progress_info" in video_data:
            vehicle_b_progress[video_data["vehicle_b_progress_info"]] += 1
        else:
            vehicle_b_progress['None'] += 1
        if "accident_negligence_rate" in video_data:
            accident_negligence[video_data["accident_negligence_rate"]] += 1
        elif "accident_negligence_rateB" in video_data:
            accident_negligence[video_data["accident_negligence_rateB"]] += 1
        else:
            accident_negligence['None'] += 1
    return {
        "NUM_ACCIDENT_TYPES": dict(accident_types),
        "NUM_ACCIDENT_PLACES": dict(accident_places),
        "NUM_ACCIDENT_PLACE_FEATURES": dict(accident_place_features),
        "NUM_VEHICLE_A_PROGRESS_INFO": dict(vehicle_a_progress),
        "NUM_VEHICLE_B_PROGRESS_INFO": dict(vehicle_b_progress),
        "NUM_NEGLIGENCE": dict(accident_negligence)
    }

def save_class_counts(annotation_dir, output_file="class_counts.json"):
    counts = get_unique_class_counts(annotation_dir)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(counts, f, indent=4)
    print(f"Class counts saved to {output_file}")

if __name__ == "__main__":
    annotation_dir = "video_datasets/train/annotation"
    save_class_counts(annotation_dir)
