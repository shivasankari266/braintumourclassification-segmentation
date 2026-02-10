import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import os

# -------------------- HYBRID CLASSIFIER MODEL --------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridModel, self).__init__()
        # Pretrained backbones
        self.resnet = create_model('resnet18', pretrained=True, num_classes=0, global_pool='avg')
        self.vit = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='avg')
        self.swin = create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=0, global_pool='avg')

        # Freeze backbones
        for param in self.resnet.parameters(): param.requires_grad = False
        for param in self.vit.parameters(): param.requires_grad = False
        for param in self.swin.parameters(): param.requires_grad = False

        # Combined feature size
        total_features = self.resnet.num_features + self.vit.num_features + self.swin.num_features
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x_swin = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        f_resnet = self.resnet(x)
        f_vit = self.vit(x)
        f_swin = self.swin(x_swin)
        features = torch.cat([f_resnet, f_vit, f_swin], dim=1)
        return self.classifier(features)


# -------------------- LOAD CLASSIFIER --------------------
def load_model(path):
    # ✅ Correct class order from training dataset
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

    model = HybridModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model, class_names


# -------------------- CLASSIFICATION PREDICTION --------------------
def predict_image(model, image_path, class_names):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
        pred_index = int(np.argmax(probs))

    return class_names[pred_index], float(probs[pred_index]), probs.tolist()


# -------------------- U-NET SEGMENTATION MODEL --------------------
def load_unet_model(path):
    model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


# -------------------- SEGMENTATION PREDICTION --------------------
def segment_and_analyze(model, image_path):
    original = Image.open(image_path).convert("RGB")
    img_np = np.array(original.resize((256, 256))) / 255.0
    img_tensor = torch.tensor(img_np.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(img_tensor)
        mask = torch.sigmoid(pred).squeeze().numpy()
        mask = (mask > 0.5).astype(np.uint8)

    tumor_area = int(np.sum(mask))

    ys, xs = np.where(mask == 1)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox = (x_min, y_min, x_max, y_max)
    else:
        bbox = None

    location_label = "No tumor detected"
    if bbox:
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2

        if y_center < 85:
            vert = "Top"
        elif y_center > 170:
            vert = "Bottom"
        else:
            vert = "Center"

        if x_center < 85:
            horiz = "Left"
        elif x_center > 170:
            horiz = "Right"
        else:
            horiz = "Center"

        location_label = f"{vert}-{horiz}"

    pixel_spacing_mm = 0.5
    area_mm2 = tumor_area * (pixel_spacing_mm ** 2)
    area_cm2 = area_mm2 / 100

    if tumor_area < 300:
        bbox = None
        location_label = "No tumor detected"
        area_cm2 = 0.0

    overlay = img_np.copy()
    overlay[mask == 1] = [1.0, 0, 0]  # red overlay
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_path = os.path.join("static/uploads", "segmented.png")
    overlay_img.save(overlay_path)

    return tumor_area, area_cm2, bbox, overlay_path, location_label
