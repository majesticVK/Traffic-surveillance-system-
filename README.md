# Traffic Surveillance AI System

## Real-time traffic monitoring using YOLOv8 + LLM (TinyLlama + LoRA)

<p align="center"> <img src="https://img.shields.io/badge/YOLOv8-Detection-blue"> <img src="https://img.shields.io/badge/LLM-TinyLlama-purple"> <img src="https://img.shields.io/badge/LoRA-Finetuned-green"> <img src="https://img.shields.io/badge/Streamlit-UI-red"> <img src="https://img.shields.io/badge/Status-Active-success"> </p>


### An intelligent traffic analysis system that combines:

1.  Computer Vision (YOLOv8) → Detect vehicles & traffic signals
2.  Rule-Based Engine → Identify congestion & violations
3.  LLM (TinyLlama + LoRA) → Human-like traffic insights
4.  Memory System → Stores and recalls past events
5.  Streamlit UI → Real-time interactive dashboard

### Because just detecting cars wasn’t ambitious enough.

## Key Features

1. ✔️ Real-Time Vehicle Detection
2. ✔️ Traffic Light State Recognition (R/Y/G)
3. ✔️ Congestion & Violation Analysis
4. ✔️ Event Logging with Memory System
5. ✔️ Chat with AI about Traffic Conditions
6. ✔️ Webcam + Video Upload Support
7. ✔️ Futuristic Dashboard UI

System Architecture

```Video Input → YOLO Detection → Rule Engine → Event Memory → LLM → UI Dashboard```

## Project Structure
```
📦 traffic-ai
├── 🚀 app.py                     # Streamlit app
├── ⚙️ config.py                  # Configurations
├── 📦 requirements.txt

├── 🧠 classification/
│   ├── traffic_detector.py       # YOLO detection
│   └── traffic_classifier.py     # Rule engine

├── 🤖 agents/
│   └── traffic_agent.py          # LLM + memory interface

├── 🗂️ memory/
│   └── event_memory.py           # Event storage

├── 🏋️ training/
│   ├── train_traffic_yolo.py     # YOLO training
│   ├── lora_finetune_traffic.py  # LLM LoRA training
│   ├── prep_traffic_dataset.py   # Dataset generation
│   └── data.py                   # updading the resolution per frames to generate a clean dataset 
```
## ⚙️ Installation
~~~
git clone https://github.com/your-username/traffic-ai.git
cd traffic-ai
pip install -r requirements.txt
~~~

Yes, it’s heavy. You signed up for AI, not minimalism.

## Run the App

```streamlit run app.py```

Open:
``` http://localhost:8501```

####  Input Modes
####  Webcam Mode → Live monitoring
####  Upload Mode → Analyze MP4 videos


## How It Works
1. ###  Detection
   Finetunned - YOLOv8 detects:
   Vehicles
   Traffic lights
   <p align="center">
  <img src="images/Screenshot_2026-03-28_020119.png
   " width="700"/>
   </p>

2. ### Classification
   Rules determine:
   NORMAL
   CONGESTED
   VIOLATION
   HIGH_RISK
   <p align="center">
  <img src="images/Screenshot_2026-03-28_015812.png
   " width="700"/>
   </p>

   
4. ###  Memory System
   Stores traffic events for context-aware reasoning

5. ###  AI Assistant
   TinyLlama + LoRA generates:
   Insights
   Explanations
   Responses

6. ###  UI Layer
   Streamlit integrates everything into a dashboard

   

8. ### Training
   YOLOv8 Fine-tuning
   python training/train_traffic_yolo.py

## Update:

```YOLO_TRAFFIC_MODEL = "path/to/best.pt"```

## LLM Fine-tuning (LoRA)

```
python training/prep_traffic_dataset.py --synthetic 20000
python training/lora_finetune_traffic.py
```


## Update:
```LORA_PATH = "./lora_traffic_final"```


## ⚙️ Configuration

Edit in config.py:
```
Speed thresholds
Model paths
Camera source
Detection sensitivity
```

## Tech Stack
```
PyTorch
Ultralytics YOLOv8
Transformers
OpenCV
Streamlit
```


## Limitations
1. Traffic light detection is HSV-based (lighting sensitive)
2. No real speed estimation yet
3. Plate detection is approximate
4. Works best on clear footage


##  Future Improvements
1. Real speed estimation (optical flow)
2. License plate recognition (OCR + detection)
3. Multi-camera tracking
4. Edge deployment (Jetson / Raspberry Pi)
5. Mobile app

## video 
https://youtu.be/pKhUgvkJ9yM

## Author

### Vansh Kumar
### AI + Vision Systems 
