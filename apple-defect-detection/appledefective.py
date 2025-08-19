import os
import cv2
import time
import io
import numpy as np
import traceback
from fastapi import FastAPI
import base64
import uvicorn
from warnings import filterwarnings
import signal
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()
from pydantic import BaseModel
from openvino.runtime import Core, PartialShape, get_version
# === Configuration ===
MODEL_PATH = r"best_model\best.xml"  # Your OpenVINO .xml model path
NUM_PARTS = 4
CLASS_LABELS = ["defective", "missing", "ok"]
class OpenVINOClassifier:
    def __init__(self, model_path: str, device: str = "GPU 0", class_labels=None):
        self.core = Core()
        available_devices = self.core.available_devices
        print(f"OpenVINO version: {get_version()}")
        print(f"Available devices: {available_devices}")

        if device.startswith("GPU 0") and not any(d.startswith("GPU 0") for d in available_devices):
            print("WARNING: GPU device requested but not available. Falling back to CPU.")
            device = "CPU"
        self.model = self.core.read_model(model_path)
        self.target_height = 224
        self.target_width = 224
        self.channels = 3

        if self.model.inputs[0].partial_shape.is_dynamic:
            self.model.reshape({self.model.inputs[0]: PartialShape([1, self.channels, self.target_height, self.target_width])})

        if device.startswith("GPU 0"):
            gpu_config = {
                "CACHE_DIR": "./gpu_cache",
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "FP32"
            }
            self.compiled_model = self.core.compile_model(self.model, device, gpu_config)
        else:
            self.compiled_model = self.core.compile_model(self.model, device)

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.class_labels = class_labels or ["defective", "missing", "ok"]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None")
        resized = cv2.resize(image, (self.target_width, self.target_height))
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return batch
    def predict(self, image: np.ndarray):
        try:
            preprocessed = self.preprocess_image(image)
            result = self.compiled_model({self.input_layer.any_name: preprocessed})
            predictions = result[self.output_layer][0]
            class_index = int(np.argmax(predictions))
            confidence = float(predictions[class_index])
            predicted_class = self.class_labels[class_index]
            return predicted_class, confidence
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            traceback.print_exc()
            return "unknown", 0.0

classifier = OpenVINOClassifier(model_path=MODEL_PATH, class_labels=CLASS_LABELS)
imgtemp1=cv2.imread(r'temp1.bmp')
imgtemp2=cv2.imread(r'temp2.bmp')
label_class1, confidence1 = classifier.predict(imgtemp1) 
label_class2, confidence2 = classifier.predict(imgtemp2) 
print("\n🔍 Inference Results:")
print(f"  📌 temp1.bmp ➤ Class: {label_class1} | Confidence: {confidence1:.2f}")
print(f"  📌 temp2.bmp ➤ Class: {label_class2} | Confidence: {confidence2:.2f}")
print("\n✅ Model loading & defect detection completed.\n")  
def process_single_image(img):
    height, width = img.shape[:2]
    part_width = width // NUM_PARTS
    anticlockwise_indices = list(reversed(range(NUM_PARTS)))
    results_list = []
    for idx, part_index in enumerate(anticlockwise_indices):
        x_start = part_index * part_width
        x_end = (part_index + 1) * part_width if part_index < NUM_PARTS - 1 else width
        part_img = img[:, x_start:x_end]
        label_class, confidence = classifier.predict(part_img)
        results_list.append({
            "id": idx + 1,
            "class": label_class,
            "confidence": round(confidence, 4)
        })
    return results_list

def string_to_image(base64_string: str) -> np.ndarray:
    imgdata = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(imgdata)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

app = FastAPI()
class ImageRequest(BaseModel):
    image: str
def opencv_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def shutdown_server():
    os.kill(os.getpid(), signal.SIGINT)

@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return {"message": "Server is shutting down..."}

@app.get("/ServerCheck")
async def server_check():
    return {"Server": "OK"}

@app.get("/")
async def home():
    return {"Health": "OK"}
@app.post("/result")
async def result(request: ImageRequest):
    start_time = time.time()
    # Decode base64 and convert to OpenCV format
    img_from_base64 = string_to_image(request.image)
    numpy_image = np.array(img_from_base64)
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    # Run image processing in a separate thread
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_single_image, image)
    inference_time_ms = (time.time() - start_time) * 1000
    print("inference_time_ms", inference_time_ms)
    print("result", result)
    return result
if __name__ == "__main__":
    uvicorn.run(
        "appledefective:app",  # Reference to the FastAPI app object in charger_api.py
        host="127.0.0.1",
        port=5001)  # ,workers=4