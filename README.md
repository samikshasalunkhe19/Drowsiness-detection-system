# 🚗 AI-Powered Real-Time Driver Drowsiness Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br>

<img src="images/demo_banner.png" width="900"/>

### Real-Time AI-Based Driver Fatigue Monitoring using Computer Vision

</div>

---

# 📌 Project Overview

Driver drowsiness is one of the major causes of road accidents worldwide.  
This project is an AI-powered real-time driver monitoring system that detects fatigue using:

- 👁️ Eye closure detection
- 😮 Yawn detection
- 🧠 Head pose estimation
- 📊 AI fatigue scoring

The system continuously monitors the driver's face using a webcam and instantly triggers alerts when signs of drowsiness are detected.

---

# ✨ Features

## 🔍 Real-Time Detection

- 👁️ Eye Aspect Ratio (EAR) based eye closure detection
- 😮 Mouth Aspect Ratio (MAR) based yawn detection
- 🧠 Head tilt & pose estimation
- 📊 AI-based fatigue score calculation
- ⚡ Real-time webcam processing

---

## 🔔 Smart Alert System

- 🔊 Audio alarm alert
- 📸 Automatic screenshot capture
- 📧 Optional email notification support
- 📝 JSON event logging system

---

# 🖥️ Dashboard Features

## 📌 KPI Summary Cards

Displays real-time summary statistics at the top of the dashboard:

| Metric | Description |
|--------|-------------|
| Total Events | Total number of logged detection events |
| Drowsiness Alerts | Number of drowsiness alerts triggered |
| Yawn Events | Total detected yawn events |
| Awake Recoveries | Times driver recovered from fatigue |

---

# 📈 Event Timeline Visualization

Interactive scatter plot displaying all recorded events over time.

### Event Categories

| Event Type | Description |
|------------|-------------|
| 🔴 DROWSINESS_DETECTED | Driver fatigue detected |
| 🟡 YAWN_DETECTED | Yawn event detected |
| 🟢 DRIVER_AWAKE | Driver recovered / alert ended |

### Features

- Interactive zoom & hover
- Color-coded event tracking
- Chronological fatigue analysis

---

# 👁️ EAR Trend Analysis

A dynamic line graph visualizing:

- Eye Aspect Ratio (EAR) values
- Fatigue progression over time
- Drowsiness threshold crossing

### Threshold Indicator

A dashed red reference line is displayed at:

```text
EAR = 0.25
```

Values below this threshold indicate prolonged eye closure and possible fatigue.

---

# ⏰ Hourly Drowsiness Frequency

A bar-chart heatmap showing which hours of the day experienced the highest number of drowsiness alerts.

### Benefits

- Identifies high-risk driving periods
- Useful for fatigue pattern analysis
- Helps evaluate driver alertness trends

---

# 📸 Screenshot Gallery

Displays all automatically captured screenshots from drowsiness events.

### Gallery Includes

- Alert images
- Timestamps
- Event filenames
- Grid-based image layout

All screenshots are loaded directly from:

```text
screenshots/
```

---

# 📋 Raw Event Log Viewer

Expandable table displaying the complete event history from:

```text
drowsiness_log.json
```

### Logged Data Includes

- Timestamp
- Event type
- EAR value
- MAR value
- Fatigue score
- Alert status

Events are automatically sorted from newest to oldest.

---

# 🔄 Auto Refresh Support

The dashboard supports:

- ✅ Manual refresh button
- ✅ 5-second live auto-refresh
- ✅ Real-time monitoring support

This allows live visualization while the detection system is actively running.

---
---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Computer vision processing |
| Dlib | Face landmark detection |
| SciPy | Distance calculations |
| NumPy | Numerical computations |
| Pygame | Alarm sound system |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-driver-drowsiness-detection.git

cd ai-driver-drowsiness-detection
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Download Dlib Facial Landmark Model

Download:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract the file and place:

```text
shape_predictor_68_face_landmarks.dat
```

inside the project folder.

---

## 4️⃣ Add Alarm Sound

Place a `.wav` file in the root folder named:

```text
alarm.wav
```

---

## 5️⃣ Run the Real-Time Detection System

```bash
python realtime_drowsiness_detection.py
```

---

## 6️⃣ Launch Streamlit Dashboard

```bash
streamlit run streamlit_dashboard.py
```

---

# 🔬 How It Works

## 👁️ Eye Aspect Ratio (EAR)

The system calculates the Eye Aspect Ratio using facial landmarks.

### Formula

\[
EAR = \frac{||p2-p6|| + ||p3-p5||}{2 \times ||p1-p4||}
\]

### Detection Logic

| EAR Value | Eye State |
|-----------|-----------|
| 0.30 – 0.40 | Open |
| 0.25 – 0.30 | Partially Closed |
| < 0.25 | Drowsy |

If EAR remains below threshold for consecutive frames, the system triggers an alert.

---

# 📊 Fatigue Score Calculation

```text
Fatigue Score =
(Eye Closure × 40)
+ (Yawn Frequency × 30)
+ (Head Pose × 20)
+ (Blink Rate × 10)
```

| Score Range | Status |
|-------------|--------|
| 0 – 24 | 🟢 Alert |
| 25 – 54 | 🟡 Moderate |
| 55 – 74 | 🟠 High |
| 75 – 100 | 🔴 Critical |

---

# ⌨️ Keyboard Controls

| Key | Function |
|-----|----------|
| q | Quit application |
| s | Save screenshot |
| d | Toggle dark/light mode |

---

# 🔮 Future Enhancements

- ✅ Deep learning fatigue classifier
- ✅ Mobile camera integration
- ✅ Flask/FastAPI deployment
- ✅ Cloud deployment on AWS/GCP
- ✅ Voice assistant alerts
- ✅ GPU acceleration support

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Samiksha Salunkhe

- 🔗 GitHub: https://github.com/samikshasalunkhe19
- 💼 LinkedIn: https://linkedin.com/in/www.linkedin.com/in/samiksha-salunkhe-03b416332

---

# ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub.

It motivates further development and helps others discover the project.

---

<div align="center">

### Built with ❤️ using Python, OpenCV, and Dlib

</div>
