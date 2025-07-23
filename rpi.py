import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# ---- Load TFLite detection model ----
MODEL_PATH = "C:/Users/Nilen Patel/Desktop/wallcrack.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <title>Wall Crack Detector</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { background:#111; text-align:center; margin:0; font-family:sans-serif; }
    h2 { color:#fff; margin-top:32px; }
    video, canvas { width:90vw; max-width:480px; border:3px solid #444; border-radius:12px; margin:16px auto; background:#222; }
    #detectBtn { margin: 18px 0 12px 0; padding: 12px 34px; font-size: 18px; color: #fff; background: #FF5722; border:none; border-radius:8px; cursor:pointer; box-shadow:0 2px 8px #0006; }
    #detectBtn:active { background:#C62828; }
    #result { color:#fff; font-size:18px; margin-top:12px; min-height:28px;}
  </style>
</head>
<body>
  <h2>Wall Crack Detector</h2>
  <video id="video" autoplay playsinline></video>
  <canvas id="canvas" style="display:none;"></canvas>
  <div>
    <button id="detectBtn">Detect Crack</button>
  </div>
  <div id="result"></div>
  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const detectBtn = document.getElementById('detectBtn');
    const resultDiv = document.getElementById('result');

    // Open camera
    navigator.mediaDevices.getUserMedia({video:true, audio:false})
      .then(stream => {
        video.srcObject = stream;
      })
      .catch(e => { resultDiv.innerText = 'Camera error: ' + e; });

    detectBtn.onclick = () => {
      resultDiv.innerText = "Detecting...";
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      canvas.toBlob(blob => {
        let form = new FormData();
        form.append('image', blob, 'frame.jpg');
        fetch('/detect', { method: 'POST', body: form })
          .then(r => r.json())
          .then(data => {
            if (!data or !Array.isArray(data) or data.length === 0) {
              resultDiv.innerText = "No crack detected.";
            } else {
              let txt = "";
              data.forEach((item, i) => {
                let box = item.box.map(n=>Math.round(n*100)/100).join(', ');
                txt += `Crack ${i+1}: Score ${(item.score*100).toFixed(1)}%, Box [${box}]<br/>`
              });
              resultDiv.innerHTML = txt;
            }
          })
          .catch(err => resultDiv.innerText = 'Error: ' + err);
      }, 'image/jpeg');
    };
  </script>
</body>
</html>
""")

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    img_file = request.files["image"]
    img = Image.open(img_file).convert("RGB")
    h_model, w_model = input_details[0]['shape'][1:3]
    img = img.resize((w_model, h_model))
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_np, 0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Adapt output parsing for your model
    # Typical SSD output: boxes, classes, scores, count
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0] if len(output_details) > 1 else None
    scores = interpreter.get_tensor(output_details[2]['index'])[0] if len(output_details) > 2 else None

    result = []
    if scores is not None and boxes is not None:
        for i, score in enumerate(scores):
            if score > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                result.append({
                    "score": float(score),
                    "box": [float(xmin), float(ymin), float(xmax), float(ymax)]
                })
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)