"""
Phase 2: Weapon Detection CNN Models
===================================

This module implements the CustomWeaponCNN_v2 architecture from Phase 2 notebook
for weapon detection (handgun, machine_gun, no_gun classification).

Based on the Phase_2_Custom_CNN_Weapon_Detection.ipynb notebook.
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

class WeaponCNNDetector:
    """
    Weapon detection classifier using CustomWeaponCNN_v2
    """
    
    def __init__(self, model_path, device='cpu'):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.class_names = ['handgun', 'machine_gun', 'no_gun']
        
        # Transforms for inference (640x640 as per Phase 2 notebook)
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the weapon detection model"""
        try:
            logger.info(f"🔫 Loading Weapon CNN from {self.model_path}")
            
            # Create model with matching architecture
            self.model = CustomWeaponCNN_v2(num_classes=3)
            
            if self.model_path.exists():
                # Load weights with weights_only=False
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Try direct loading
                        self.model.load_state_dict(checkpoint)
                else:
                    # Direct model weights
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"✅ Weapon CNN loaded successfully")
                logger.info(f"📊 Model size: {self.model.get_model_size():.2f} MB")
                logger.info(f"🎯 Classes: {self.class_names}")
                
            else:
                logger.warning(f"⚠️ Weapon model not found: {self.model_path}")
                logger.info("Using placeholder weapon detector")
                
        except Exception as e:
            logger.error(f"❌ Failed to load weapon model: {e}")
            logger.info("Using placeholder weapon detector")
    
    def classify(self, image):
        """
        Classify weapon in image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with classification result
        """
        if self.model is None:
            # Placeholder result
            return {
                'class': 'no_gun',
                'confidence': 0.5,
                'class_id': 2,
                'all_scores': [0.1, 0.1, 0.8]
            }
        
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1)
                scores = probs.cpu().numpy()[0]
                
                predicted_class_id = np.argmax(scores)
                predicted_class = self.class_names[predicted_class_id]
                confidence = float(scores[predicted_class_id])
                
                return {
                    'class': predicted_class,
                    'confidence': confidence,
                    'class_id': predicted_class_id,
                    'all_scores': scores.tolist()
                }
                
        except Exception as e:
            logger.error(f"❌ Error in weapon classification: {e}")
            return {
                'class': 'no_gun',
                'confidence': 0.0,
                'class_id': 2,
                'all_scores': [0.0, 0.0, 1.0]
            }
    
    def detect_weapons_in_detections(self, detections, frames):
        """
        Add weapon classification to existing person detections
        
        Args:
            detections: List of detection dictionaries
            frames: List of frame arrays
            
        Returns:
            Enhanced detections with weapon information
        """
        enhanced_detections = []
        
        for detection in detections:
            enhanced_detection = detection.copy()
            
            # Only classify if it's a person detection
            if detection.get('label') == 'person':
                try:
                    # Get frame for this detection
                    frame_idx = detection.get('frame_index', 0)
                    if frame_idx < len(frames):
                        frame = frames[frame_idx]
                        
                        # Crop person region
                        bbox = detection.get('bbox_xyxy', [0, 0, frame.shape[1], frame.shape[0]])
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        
                        # Ensure valid crop region
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            person_crop = frame[y1:y2, x1:x2]
                            
                            # Classify weapon
                            weapon_result = self.classify(person_crop)
                            enhanced_detection['weapon_classification'] = weapon_result
                        else:
                            # Invalid crop, use default
                            enhanced_detection['weapon_classification'] = {
                                'class': 'no_gun',
                                'confidence': 0.0,
                                'class_id': 2,
                                'all_scores': [0.0, 0.0, 1.0]
                            }
                            
                except Exception as e:
                    logger.error(f"❌ Error classifying weapon for detection: {e}")
                    enhanced_detection['weapon_classification'] = {
                        'class': 'no_gun', 
                        'confidence': 0.0,
                        'class_id': 2,
                        'all_scores': [0.0, 0.0, 1.0]
                    }
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections