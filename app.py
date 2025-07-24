from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import os
import logging
import gc
import signal
import resource
import psutil
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration Constants
MODEL_NAME = "Claudineuwa/waste_classifier_Isaac"
MAX_IMAGES_PER_REQUEST = 5  # Limit to prevent memory overload
MEMORY_LIMIT_PERCENTAGE = 0.8  # Use 80% of available memory

# Type aliases
PredictionResult = Dict[str, Any]

# Label information
LABEL2INFO = {
    0: {
        "label": "biodegradable",
        "description": "Easily breaks down naturally. Good for composting.",
        "recyclable": False,
        "disposal": "Use compost or organic bin",
        "example_items": ["banana peel", "food waste", "paper"],
        "environmental_benefit": "Composting biodegradable waste returns nutrients to the soil.",
        "protection_tip": "Compost at home or use municipal organic waste bins.",
        "poor_disposal_effects": "Can cause methane emissions in landfills."
    },
    1: {
        "label": "non_biodegradable",
        "description": "Does not break down easily. Should be disposed of carefully.",
        "recyclable": False,
        "disposal": "Use general waste bin or recycling if possible",
        "example_items": ["plastic bag", "styrofoam", "metal can"],
        "environmental_benefit": "Proper disposal reduces pollution and protects wildlife.",
        "protection_tip": "Reduce use, reuse items, and recycle whenever possible.",
        "poor_disposal_effects": "Leads to soil and water pollution, harms wildlife."
    }
}

# Global variables for model and processor
model: Any = None
image_processor: Any = None

def set_memory_limit() -> None:
    """Set memory limits to prevent OOM errors"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        total_mem = psutil.virtual_memory().total
        new_limit = int(total_mem * MEMORY_LIMIT_PERCENTAGE)
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
        logger.info(f"Set memory limit to {new_limit/1024/1024:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not set memory limits: {e}")

def cleanup(signum=None, frame=None) -> None:
    """Handle signals to clean up memory"""
    global model, image_processor
    
    logger.info("Performing memory cleanup...")
    if model is not None:
        model.cpu()
        del model
    if image_processor is not None:
        del image_processor
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    logger.info("Cleanup completed")

def load_model() -> bool:
    """Load the model with memory optimization"""
    global model, image_processor
    
    # Clean up existing resources
    cleanup()
    
    try:
        logger.info(f"Loading model {MODEL_NAME} with memory optimization...")
        
        # Load with aggressive memory settings
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            offload_folder="offload",
            offload_state_dict=True
        )
        
        # Load processor
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        
        # Set to eval mode and move to GPU if available
        model.eval()
        if torch.cuda.is_available():
            model = model.to('cuda')
            
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        cleanup()
        return False

def predict_image(image_bytes: bytes, device: str = "cpu") -> PredictionResult:
    """Predict image classification with memory management"""
    if model is None or image_processor is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Use context manager for automatic cleanup
        with torch.inference_mode():
            # Process image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = image_processor(images=image, return_tensors="pt")
            
            # Move only necessary tensors to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, pred = torch.max(probs, dim=1)
            
            # Convert to CPU numpy immediately
            label_id = pred.cpu().numpy()[0]
            confidence = conf.cpu().item()
            
            # Get result
            result = LABEL2INFO[label_id].copy()
            result["confidence"] = round(confidence, 2)
            return result
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise
    finally:
        # Ensure cleanup even if error occurs
        del inputs, outputs, probs, conf, pred if 'inputs' in locals() else None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "memory": {
            "available": psutil.virtual_memory().available / (1024**2),
            "used": psutil.virtual_memory().used / (1024**2),
            "percent": psutil.virtual_memory().percent
        }
    })

@app.route('/classify', methods=['POST'])
def classify() -> Any:
    """Classification endpoint with memory management"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Limit number of images processed
        files = request.files.getlist('images')[:MAX_IMAGES_PER_REQUEST]
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        results = []
        for file in files:
            if not file.filename:
                continue
                
            try:
                image_bytes = file.read()
                result = predict_image(image_bytes)
                results.append(result)
                
                # Immediate cleanup
                del image_bytes
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                continue
                
        return jsonify({
            "results": results,
            "processed_count": len(results),
            "warning": f"Limited to first {MAX_IMAGES_PER_REQUEST} images" if len(request.files.getlist('images')) > MAX_IMAGES_PER_REQUEST else None
        })
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"error": "Processing error"}), 500
    finally:
        cleanup()

@app.route('/memory-status', methods=['GET'])
def memory_status() -> Dict[str, Any]:
    """Detailed memory status endpoint"""
    mem = psutil.virtual_memory()
    return jsonify({
        "total_memory_gb": round(mem.total / (1024**3), 2),
        "available_memory_gb": round(mem.available / (1024**3), 2),
        "used_memory_percent": mem.percent,
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory": torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    })

# Register signal handlers
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)

# Initialize
if __name__ == '__main__':
    try:
        set_memory_limit()
        if not load_model():
            logger.error("Failed to load model - exiting")
            exit(1)
            
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        cleanup()
        exit(1)