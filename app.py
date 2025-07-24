import os
import gc
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = AutoImageProcessor.from_pretrained("Isaac-1-lang/waste_classifier_Isaac")
model = AutoModelForImageClassification.from_pretrained("Isaac-1-lang/waste_classifier_Isaac").to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Label mapping
id2label = model.config.id2label

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        image = Image.open(file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        result = id2label[pred.item()]
        return jsonify({
            "prediction": result,
            "confidence": f"{conf.item():.4f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Safe cleanup
        for var in ['inputs', 'outputs', 'probs', 'conf', 'pred']:
            if var in locals():
                del locals()[var]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
