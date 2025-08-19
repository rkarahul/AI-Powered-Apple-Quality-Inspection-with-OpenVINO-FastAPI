
# 🍏 AI-Powered Apple Quality Inspection with OpenVINO & FastAPI  

## 📖 Overview  
This project is an **AI-based apple defect detection system** that uses **OpenVINO** for optimized inference and **FastAPI** for REST API deployment.  
It classifies apple images into categories: **Defective**, **Missing**, and **OK**, and splits each image into **4 parts** for localized inspection.  
The API can also return the **output image with classification results** (labels + confidence) as a Base64-encoded string.  

![24_07_01_13_39_27_classified](https://github.com/user-attachments/assets/155f8fc0-a390-489e-b24a-400450809983)
![24_07_01_13_53_70_classified](https://github.com/user-attachments/assets/0b254ea2-6749-462a-abac-88fdf7ea01b2)



## ✨ Features  
- ✅ OpenVINO-optimized inference (CPU/GPU support)  
- ✅ Multi-part apple inspection (split into 4 regions)  
- ✅ REST API with FastAPI  
- ✅ Base64 input/output support  
- ✅ Confidence scores per classification  
- ✅ Option to **return annotated output image**  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **OpenVINO Runtime**  
- **FastAPI**  
- **Uvicorn**  
- **OpenCV**  
- **NumPy**  
- **Pillow (PIL)**  

---

## 📂 Project Structure  
```
apple-defect-detection/
│── best_model/                # OpenVINO IR model (.xml, .bin)
│── appledefective.py          # FastAPI application
│── temp1.bmp                  # Sample input image
│── temp2.bmp                  # Sample input image
│── requirements.txt           # Dependencies
│── README.md                  # Documentation
```

---

## ⚡ Installation  

### 1. Clone Repository  
```bash
git clone https://github.com/yourusername/apple-defect-detection.git
cd apple-defect-detection
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Place Model Files  
Put your OpenVINO `.xml` and `.bin` files inside the `best_model/` folder.  

---

## 🚀 Running the Server  
```bash
python appledefective.py
```
The API will start at:  
```
http://127.0.0.1:5001
```

---

## 📡 API Endpoints  

### 🔹 Health Check  
```http
GET /ServerCheck
```
**Response:**  
```json
{"Server": "OK"}
```

### 🔹 Predict Apple Image (Defect Detection)  
```http
POST /result
```

**Request Body (Base64 Image):**  
```json
{
  "image": "BASE64_STRING_HERE"
}
```

**Response Example (JSON + Annotated Image):**  
```json
{
  "results": [
    {"id": 1, "class": "ok", "confidence": 0.9532},
    {"id": 2, "class": "defective", "confidence": 0.8431},
    {"id": 3, "class": "ok", "confidence": 0.9724},
    {"id": 4, "class": "missing", "confidence": 0.6655}
  ],
  "output_image": "BASE64_STRING_OF_ANNOTATED_IMAGE"
}
```

---

## 🖼️ Example Workflow  
1. Capture apple image  
2. Convert image → **Base64**  
3. Send to **API** (`/result`)  
4. Get back:  
   - **Classification results (JSON)**  
   - **Annotated output image** (Base64 string)  

---

## 🚀 Future Improvements  
- Add **Docker deployment**  
- Support **industrial cameras (FLIR, Basler)**  
- Save results into **database**  
- Deploy on **edge devices**  

---

## 📜 License  
MIT License  
