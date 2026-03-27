YOLO_MODEL = "yolov8n.pt"              # base model; fine-tuned weights override this
YOLO_TRAFFIC_MODEL = "traffic_yolo_finetuned.pt"  # your fine-tuned model (after training)
EVENT_DB = "traffic_events.json"

# Traffic classifier thresholds
SPEED_LIMIT_KMH = 60                   # used for speed-violation scoring
CONGESTION_VEHICLE_COUNT = 5           # vehicles in frame → congestion flag
RED_LIGHT_CONF_THRESHOLD = 0.75        # confidence to call a red-light violation

# Vehicle classes YOLOv8 detects (COCO)
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
VIOLATION_OBJECTS = ["traffic light"]  # objects that trigger violation checks

# Traffic light states (from your custom detector or classifier)
TRAFFIC_LIGHT_STATES = ["red", "green", "yellow"]

# LLM settings
BASE_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "./lora_traffic_final"     # set after training

# Camera
CAMERA_ID = 0                          # webcam or dashcam index; use RTSP string for IP cam
# CAMERA_ID = "rtsp://192.168.1.x:554/stream"

# Map / GPS (optional — attach GPS dongle and read NMEA)
GPS_ENABLED = False
DEFAULT_LOCATION = "Unknown road"
