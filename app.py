from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_image, load_unet_model, segment_and_analyze
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------- Flask App --------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# -------------------- Load Models --------------------
classifier, class_names = load_model('brain_tumor_hybrid_model.pth')
segmentor = load_unet_model('tumor_segmentation_unet.pth')

# -------------------- Label Mapping --------------------
LABEL_MAP = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor Detected"
}

# -------------------- Tumor Information --------------------
condition_info = {
    "Glioma Tumor": {
        "description": "Gliomas originate from glial cells and can be aggressive.",
        "cause": "Genetic mutations, radiation exposure.",
        "advice": "Consult a neuro-oncologist. Treatment may include surgery, radiation, and chemotherapy.",
        "map_query": "glioma brain tumor treatment hospitals near me"
    },
    "Meningioma Tumor": {
        "description": "Meningiomas arise from brain membranes and are often benign.",
        "cause": "Radiation exposure or hormonal factors.",
        "advice": "Consult a neurosurgeon for monitoring or surgical removal.",
        "map_query": "meningioma tumor treatment hospitals near me"
    },
    "Pituitary Tumor": {
        "description": "Tumors affecting hormone regulation in the pituitary gland.",
        "cause": "Hormonal imbalance or genetic conditions.",
        "advice": "Visit an endocrinologist. Treatment varies by type.",
        "map_query": "pituitary tumor hospital near me"
    },
    "No Tumor Detected": {
        "description": "No abnormal mass detected in the MRI.",
        "cause": "-",
        "advice": "Maintain healthy habits. Consult a neurologist if symptoms persist.",
        "map_query": "neurology health checkup near me"
    }
}

# -------------------- MRI Validation --------------------
def is_mri_image(filepath):
    try:
        img = Image.open(filepath).convert("L")
        w, h = img.size

        if w < 128 or h < 128:
            return False

        aspect = w / h
        if aspect < 0.7 or aspect > 1.3:
            return False

        hist = img.histogram()
        if sum(1 for v in hist if v > 0) < 50:
            return False

        mean_intensity = sum(i * hist[i] for i in range(256)) / sum(hist)
        if mean_intensity < 20 or mean_intensity > 230:
            return False

        return True
    except:
        return False

# -------------------- Routes --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', message="No file uploaded")

    file = request.files['file']
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ---- Step 1: Validate MRI ----
    if not is_mri_image(filepath):
        return render_template(
            'error.html',
            message="Invalid image. Please upload a valid brain MRI."
        )

    # ---- Step 2: Segmentation ----
    tumor_area, area_cm2, bbox, overlay_path, location_label = segment_and_analyze(
        segmentor, filepath
    )

    # ---- Step 3: Classification ----
    raw_prediction, confidence, probs = predict_image(
        classifier, filepath, class_names
    )

    prediction = LABEL_MAP.get(raw_prediction, raw_prediction)

    # ---- Step 4: Hybrid Validation ----
    if raw_prediction != "notumor" and confidence < 0.5 and tumor_area < 50:
        return render_template(
            'error.html',
            message="Uncertain prediction. Please upload a clearer MRI scan."
        )

    if raw_prediction == "notumor":
        prediction = "No Tumor Detected"
        confidence = max(confidence, 0.9)
        tumor_area = 0
        area_cm2 = 0.0
        bbox = None
        location_label = "N/A"

    # ---- Step 5: Confidence Donut Chart ----
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'confidence_chart.png')
    plt.figure(figsize=(5, 5))
    plt.pie(
        [confidence, 1 - confidence],
        labels=[prediction, "Other"],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    plt.text(0, 0, f"{round(confidence * 100)}%", ha='center', va='center',
             fontsize=18, fontweight='bold')
    plt.title("Prediction Confidence")
    plt.savefig(chart_path, transparent=True)
    plt.close()

    # ---- Step 6: Condition Info ----
    details = condition_info.get(prediction, {
        "description": "No information available.",
        "cause": "-",
        "advice": "Consult a specialist.",
        "map_query": "brain tumor hospital near me"
    })

    # ---- Step 7: Render Result ----
    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        image_path=filepath,
        chart_path=chart_path,
        details=details,
        tumor_area=tumor_area,
        area_cm2=area_cm2,
        bbox=bbox,
        overlay_path=overlay_path,
        location_label=location_label
    )

# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(debug=True)
