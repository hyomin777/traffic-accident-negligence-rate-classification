import torch

NEGLIGENCE_CATEGORIES = {
    "0:100": 0, 
    "10:90": 1, 
    "20:80": 2, 
    "30:70": 3, 
    "40:60": 4, 
    "50:50": 5, 
    "60:40": 6, 
    "70:30": 7, 
    "80:20": 8, 
    "90:10": 9, 
    "100:0": 10
}

OBJECT_CLASSES = {
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

CLASS_LIST = [
    'vehicle',
    'pedestrian',
    'traffic-sign',
    'traffic-light-green',
    'traffic-light-red',
    'traffic-light-etc',
    'crosswalk',
    'two-wheeled-vehicle',
    'bike'
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_OBJECT_CLASSES = 9
NUM_NEGLIGENCE_CLASSES = len(NEGLIGENCE_CATEGORIES)
MAX_DETACTIONS = 12
NUM_ACCIDENT_TYPES = 434
NUM_ACCIDENT_PLACES = 15
NUM_ACCIDENT_PLACE_FEATURES = 61
NUM_VEHICLE_A_PROGRESS_INFO = 180
NUM_VEHICLE_B_PROGRESS_INFO = 174

MAX_FRAMES = 16
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4

EPOCHS = 10
LR = 0.00005
AUX_LAMBDA = 0.12
GAMMA = 2.0
AUX_GAMMA = 1.5
