"""
Custom CNN Architecture for Phase 1 Wildlife Detection
====================================================

This module contains the CustomAnimalCNN architecture used in Phase 1 training.
It implements the EXACT SAME architecture as defined in the Phase 1 notebook for 
animal-specific detection models.

The models are trained as individual specialists and combined using ensemble voting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class DepthwiseSeparableConv(nn.Module):
    """Memory-efficient depthwise separable convolution - EXACT MATCH with Phase 1"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class AnimalDetectionHead(nn.Module):
    """Detection head for classification + bounding box regression - EXACT MATCH"""
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        
        # Classification head
        self.cls_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.cls_head = nn.Conv2d(64, num_classes, 1)
        
        # Bounding box regression head
        self.bbox_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bbox_head = nn.Conv2d(64, 4, 1)  # x, y, w, h
        
        # Objectness head
        self.obj_conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.obj_head = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        # Classification
        cls_feat = F.relu(self.cls_conv(x))
        cls_output = torch.sigmoid(self.cls_head(cls_feat))
        
        # Bounding box
        bbox_feat = F.relu(self.bbox_conv(x))
        bbox_output = self.bbox_head(bbox_feat)
        
        # Objectness
        obj_feat = F.relu(self.obj_conv(x))
        obj_output = torch.sigmoid(self.obj_head(obj_feat))
        
        return cls_output, bbox_output, obj_output

class CustomAnimalCNN(nn.Module):
    """
    Lightweight Custom CNN for Animal Detection - EXACT MATCH WITH PHASE 1 NOTEBOOK
    This architecture EXACTLY matches the Phase 1 notebook implementation
    
    Features:
    - Lightweight architecture (<5M parameters)  
    - Depthwise separable convolutions
    - Multi-scale detection head
    - Memory-efficient design
    """
    
    def __init__(self, num_classes=1, input_size=416):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone - Lightweight feature extractor (EXACT MATCH with Phase 1)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Efficient blocks using depthwise separable convolutions (EXACT MATCH)
        self.blocks = nn.ModuleList([
            DepthwiseSeparableConv(32, 64, stride=2),    # 208x208
            DepthwiseSeparableConv(64, 128, stride=2),   # 104x104
            DepthwiseSeparableConv(128, 256, stride=2),  # 52x52
            DepthwiseSeparableConv(256, 512, stride=2),  # 26x26
            DepthwiseSeparableConv(512, 512, stride=1),  # 26x26
        ])
        
        # Feature Pyramid Network (FPN) - Multi-scale features (EXACT MATCH)
        self.fpn_conv = nn.Conv2d(512, 256, 1, bias=False)
        self.fpn_bn = nn.BatchNorm2d(256)
        
        # Detection head (EXACT MATCH)
        self.detection_head = AnimalDetectionHead(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Feature extraction
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        # Use the last feature map for detection
        x = F.relu(self.fpn_bn(self.fpn_conv(features[-1])))
        
        # Detection outputs
        cls_output, bbox_output, obj_output = self.detection_head(x)
        
        return {
            'classification': cls_output,
            'bbox_regression': bbox_output,
            'objectness': obj_output
        }
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    # Layer freezing methods for transfer learning
    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze backbone layers for transfer learning"""
        for param in self.stem.parameters():
            param.requires_grad = not freeze
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = not freeze
        for param in self.fpn_conv.parameters():
            param.requires_grad = not freeze
        for param in self.fpn_bn.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"🧊 Backbone layers {status} for transfer learning")
    
    def freeze_early_layers(self, num_layers=3):
        """Freeze only the early layers for fine-tuning"""
        # Freeze stem
        for param in self.stem.parameters():
            param.requires_grad = False
        
        # Freeze first num_layers blocks
        for i, block in enumerate(self.blocks):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = False
                    
        logger.info(f"🧊 First {num_layers + 1} layers frozen for fine-tuning")


# =============================================================================
# PHASE 2: WEAPON DETECTION CNN (Based on Phase 2 notebook)
# =============================================================================

class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2) - FOR WEAPON DETECTION"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Expand
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise + Pointwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class CustomWeaponCNN_v2(nn.Module):
    """
    Unified weapon detection model - EXACT MATCH with Phase 2 notebook
    Architecture: MobileNetV2-inspired with ~4.8M parameters
    Classes: handgun, machine_gun, no_gun
    """
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Backbone (MobileNetV2-style inverted residuals)
        # Input: 640x640 -> 320x320
        self.layer1 = nn.Sequential(
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),  # 160x160
            InvertedResidual(24, 24, stride=1, expand_ratio=6)
        )
        
        self.layer2 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),  # 80x80
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6)
        )
        
        self.layer3 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),  # 40x40
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6)
        )
        
        self.layer4 = nn.Sequential(
            InvertedResidual(96, 160, stride=2, expand_ratio=6),  # 20x20
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(160, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


# =============================================================================
# ENSEMBLE DETECTOR CLASS  
# =============================================================================

class CustomCNNEnsembleDetector:
    """
    Ensemble detector for Custom CNN models
    Handles loading and inference of 6 specialized animal detection models
    """
    
    def __init__(self, models_dir, device='cpu'):
        self.models_dir = Path(models_dir)
        self.device = device
        self.models = {}
        self.class_names = ['human', 'elephant', 'lion', 'hippo', 'rhino', 'crocodile']
        
        # Transforms for inference
        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all animal-specific models"""
        logger.info("🔄 Loading Custom CNN Ensemble Detector - 6 Specialized Models")
        logger.info(f"📁 Model Directory: {self.models_dir}")
        logger.info(f"🖥️ Device: {self.device}")
        
        model_files = {
            'human': self.models_dir / "human_stage1.pt",
            'elephant': self.models_dir / "elephant_stage1.pt", 
            'lion': self.models_dir / "lion_stage1.pt",
            'hippo': self.models_dir / "hippo_stage1.pt",
            'rhino': self.models_dir / "rhino_stage1.pt",
            'crocodile': self.models_dir / "crocodile_stage1.pt"
        }
        
        available_models = sum(1 for path in model_files.values() if path.exists())
        logger.info(f"📊 Model Availability: {available_models}/6 models found")
        
        for animal, model_path in model_files.items():
            try:
                if model_path.exists():
                    # Create model with matching architecture
                    model = CustomAnimalCNN(num_classes=1, input_size=416)
                    
                    # Load weights with weights_only=False (to handle YOLO saved models)
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict']) 
                        else:
                            # Try direct loading
                            model.load_state_dict(checkpoint)
                    else:
                        # Direct model weights
                        model.load_state_dict(checkpoint)
                    
                    model.to(self.device)
                    model.eval()
                    self.models[animal] = model
                    logger.info(f"✅ {animal} model loaded successfully")
                    
                else:
                    logger.warning(f"⚠️ {animal} model not found: {model_path}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {animal} model: {e}")
        
        if not self.models:
            raise ValueError("No custom CNN models found in /Users/nishchalnaithani/Desktop/venv/new_capstone/deployment_models/phase1")
    
    def detect(self, image, confidence_threshold=0.3):
        """
        Run ensemble detection on image
        
        Args:
            image: PIL Image or numpy array
            confidence_threshold: Detection confidence threshold
            
        Returns:
            List of detection dictionaries
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        detections = []
        
        with torch.no_grad():
            for animal, model in self.models.items():
                try:
                    output = model(input_tensor)
                    
                    # Extract predictions
                    classification = output['classification'].cpu().numpy()[0]
                    bbox_regression = output['bbox_regression'].cpu().numpy()[0] 
                    objectness = output['objectness'].cpu().numpy()[0]
                    
                    # Find detections above threshold
                    obj_mask = objectness[0] > confidence_threshold
                    if np.any(obj_mask):
                        # Get detection locations
                        y_indices, x_indices = np.where(obj_mask)
                        
                        for y_idx, x_idx in zip(y_indices, x_indices):
                            confidence = float(objectness[0, y_idx, x_idx])
                            
                            # Extract bounding box (relative to grid cell)
                            bbox = bbox_regression[:, y_idx, x_idx]
                            
                            # Convert to image coordinates (simplified)
                            img_w, img_h = image.size
                            grid_w, grid_h = 26, 26  # Based on model output size
                            
                            center_x = (x_idx + bbox[0]) * (img_w / grid_w)
                            center_y = (y_idx + bbox[1]) * (img_h / grid_h) 
                            width = bbox[2] * img_w
                            height = bbox[3] * img_h
                            
                            # Convert to xyxy format
                            x1 = max(0, center_x - width/2)
                            y1 = max(0, center_y - height/2)
                            x2 = min(img_w, center_x + width/2)
                            y2 = min(img_h, center_y + height/2)
                            
                            detections.append({
                                'label': animal,
                                'confidence': confidence,
                                'bbox_xyxy': [x1, y1, x2, y2],
                                'center_point': [center_x, center_y],
                                'area': width * height
                            })
                            
                except Exception as e:
                    logger.error(f"❌ Error in {animal} model inference: {e}")
        
        # Apply NMS to remove overlapping detections
        detections = self._apply_nms(detections, iou_threshold=0.5)
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [det for det in detections 
                         if self._calculate_iou(best['bbox_xyxy'], det['bbox_xyxy']) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def load_ensemble_from_best_pt(model_path: str, device: str = "mps") -> CustomCNNEnsembleDetector:
    """
    Attempt to load ensemble from best.pt file
    Falls back to individual model loading if needed
    """
    try:
        # Try to load as YOLO first (for backward compatibility)
        import torch
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        logger.info(f"✅ Loaded best.pt as ensemble checkpoint")
        
        # If it's a YOLO model, we'll need to fall back to individual loading
        if hasattr(checkpoint, 'model'):
            logger.warning("⚠️ best.pt is YOLO format, falling back to individual models")
            return None
            
        return checkpoint
    except Exception as e:
        logger.error(f"❌ Error loading best.pt: {e}")
        return None