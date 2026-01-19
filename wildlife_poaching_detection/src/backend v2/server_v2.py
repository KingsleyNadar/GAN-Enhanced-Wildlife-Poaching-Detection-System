"""
Wildlife Protection AI System - Integrated FastAPI Backend
==========================================================
Complete integration with Next.js frontend
"""

import os
import sys
import json
import time
import uuid
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

import cv2
import numpy as np
import torch
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure paths
BASE_DIR = Path(__file__).parent
DEPLOYMENT_MODELS_DIR = BASE_DIR / "deployment_models"
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# Model paths
PHASE1_WEIGHTS = DEPLOYMENT_MODELS_DIR / "phase1" / "best.pt"
PHASE2_YOLO_WEIGHTS = "yolov8n.pt"
PHASE2_WEAPON_WEIGHTS = DEPLOYMENT_MODELS_DIR / "phase2" / "phase2_detector_b_best.pt"
AUDIO_MODEL_PATH = DEPLOYMENT_MODELS_DIR / "audio" / "audio_classifier.h5"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"🔧 Using device: {DEVICE}")

# Constants
FRAME_SIZE = 640
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 5.0
MEL_BINS = 13
N_FFT = 2048
HOP_LENGTH = 512

PHASE1_CLASSES = {
    0: "human", 1: "elephant", 2: "lion", 3: "giraffe", 
    4: "dog", 5: "crocodile", 6: "hippo", 7: "zebra", 
    8: "rhino", 9: "animal_unknown"
}

AUDIO_CLASSES = ['POACHING_THREAT', 'MACHINERY', 'NATURAL']

ENDANGERED_SPECIES = {
    "elephant", "rhino", "lion", "hippo", "cheetah"
}

PERSON_LABELS = {"person", "people", "human"}
WEAPON_LABELS = {"machine_gun", "gun", "weapon", "firearm", "knife", "handgun", "shotgun", "rifle_or_gun"}
VEHICLE_LABELS = {"vehicle", "car", "truck", "bus", "van", "motorcycle", "bicycle"}
FIRE_LABELS = {"fire", "smoke", "fire_or_smoke"}

# =============================================================================
# DATA MODELS
# =============================================================================

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox_xyxy: List[float]

class FramePrediction(BaseModel):
    frame_index: int
    timestamp: float
    detections: List[DetectedObject]

class AudioPrediction(BaseModel):
    label: str
    confidence: float
    segment_duration: float = 5.0

class AlertEvidence(BaseModel):
    detection_source: str
    confidence_scores: Dict[str, float] = {}
    spatial_context: Optional[Dict[str, Any]] = None
    threat_assessment: str

class AlertPrediction(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    rule: str
    confidence: float
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    evidence: AlertEvidence
    recommended_actions: List[str] = []
    response_time_target: str = "< 15 minutes"

class GanPrediction(BaseModel):
    operation_type: str
    escalation_pattern: str
    risk_assessment: str
    confidence: float

class AnalysisResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    phase1_frames: List[FramePrediction] = []
    phase2_frames: List[FramePrediction] = []
    audio: Optional[AudioPrediction] = None
    alerts: List[AlertPrediction] = []
    gan_prediction: Optional[GanPrediction] = None
    processing_metadata: Dict[str, Any] = {}

# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    def __init__(self):
        self.phase1_model = None
        self.phase2_yolo_model = None
        self.phase2_weapon_model = None
        self.audio_model = None
        self.fusion_engine = None
        self.is_loaded = False
    
    async def load_all_models(self):
        """Load all AI models"""
        try:
            logger.info("🚀 Loading all AI models...")
            
            # Phase 1: Animal Detection
            if PHASE1_WEIGHTS.exists():
                logger.info(f"Loading Phase 1 model from {PHASE1_WEIGHTS}")
                self.phase1_model = YOLO(str(PHASE1_WEIGHTS))
                self.phase1_model.to(DEVICE)
            else:
                logger.warning(f"Phase 1 weights not found at {PHASE1_WEIGHTS}")
            
            # Phase 2: YOLOv8n for general detection
            logger.info("Loading Phase 2 YOLOv8n model")
            self.phase2_yolo_model = YOLO(PHASE2_YOLO_WEIGHTS)
            self.phase2_yolo_model.to(DEVICE)
            
            # Phase 2: Weapon CNN
            if PHASE2_WEAPON_WEIGHTS.exists():
                logger.info(f"Loading Weapon CNN from {PHASE2_WEAPON_WEIGHTS}")
                self.phase2_weapon_model = YOLO(str(PHASE2_WEAPON_WEIGHTS))
                self.phase2_weapon_model.to(DEVICE)
            else:
                logger.warning(f"Weapon model not found at {PHASE2_WEAPON_WEIGHTS}")
            
            # Audio Classifier
            if AUDIO_MODEL_PATH.exists():
                logger.info(f"Loading Audio model from {AUDIO_MODEL_PATH}")
                self.audio_model = tf.keras.models.load_model(str(AUDIO_MODEL_PATH))
            else:
                logger.warning(f"Audio model not found at {AUDIO_MODEL_PATH}")
            
            # Initialize Fusion Engine
            self.fusion_engine = SuperModelFusionEngine()
            
            self.is_loaded = True
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def process_video_phase1(self, video_path: str, frame_stride: int = 5) -> List[FramePrediction]:
        """Process video through Phase 1 (Animal Detection)"""
        if not self.phase1_model:
            logger.warning("Phase 1 model not loaded, returning empty results")
            return []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_predictions = []
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_stride == 0:
                    # Run inference
                    results = self.phase1_model(frame, conf=0.3, verbose=False)
                    
                    detections = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                xyxy = box.xyxy[0].cpu().numpy().tolist()
                                
                                label = PHASE1_CLASSES.get(cls_id, f"class_{cls_id}")
                                detections.append(DetectedObject(
                                    label=label,
                                    confidence=conf,
                                    bbox_xyxy=xyxy
                                ))
                    
                    frame_predictions.append(FramePrediction(
                        frame_index=frame_idx,
                        timestamp=frame_idx / fps,
                        detections=detections
                    ))
                
                frame_idx += 1
        
        finally:
            cap.release()
        
        logger.info(f"Phase 1: Processed {len(frame_predictions)} frames")
        return frame_predictions
    
    def process_video_phase2(self, video_path: str, frame_stride: int = 5) -> List[FramePrediction]:
        """Process video through Phase 2 (Threat Detection)"""
        if not self.phase2_yolo_model:
            logger.warning("Phase 2 model not loaded, returning empty results")
            return []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_predictions = []
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_stride == 0:
                    # Run YOLOv8n inference
                    results = self.phase2_yolo_model(frame, conf=0.3, verbose=False)
                    
                    detections = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                xyxy = box.xyxy[0].cpu().numpy().tolist()
                                
                                label = result.names[cls_id]
                                detections.append(DetectedObject(
                                    label=label,
                                    confidence=conf,
                                    bbox_xyxy=xyxy
                                ))
                    
                    # Run Weapon CNN if available
                    if self.phase2_weapon_model:
                        weapon_results = self.phase2_weapon_model(frame, conf=0.3, verbose=False)
                        for result in weapon_results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                                    
                                    weapon_label = result.names.get(cls_id, f"weapon_{cls_id}")
                                    detections.append(DetectedObject(
                                        label=weapon_label,
                                        confidence=conf,
                                        bbox_xyxy=xyxy
                                    ))
                    
                    frame_predictions.append(FramePrediction(
                        frame_index=frame_idx,
                        timestamp=frame_idx / fps,
                        detections=detections
                    ))
                
                frame_idx += 1
        
        finally:
            cap.release()
        
        logger.info(f"Phase 2: Processed {len(frame_predictions)} frames")
        return frame_predictions
    
    def process_audio(self, audio_path: str) -> Optional[AudioPrediction]:
        """
        Process audio through classifier.
        This function now accepts a path to a video file.
        librosa.load() will automatically extract the audio stream.
        """
        if not self.audio_model:
            logger.warning("Audio model not loaded")
            return None
        
        try:
            # Load audio from video file
            audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=MEL_BINS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            
            # Pad or truncate to fixed length
            target_length = 13  # Adjust based on your model
            if mfccs.shape[1] < target_length:
                pad_width = target_length - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :target_length]
            
            # Reshape for model input
            mfccs = mfccs[np.newaxis, ..., np.newaxis]
            
            # Predict
            predictions = self.audio_model.predict(mfccs, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            return AudioPrediction(
                label=AUDIO_CLASSES[class_idx],
                confidence=confidence,
                segment_duration=AUDIO_DURATION
            )
        
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None

# =============================================================================
# FUSION ENGINE
# =============================================================================

class SuperModelFusionEngine:
    """Rule-based fusion engine for generating alerts"""
    
    def __init__(self):
        self.endangered_classes = ENDANGERED_SPECIES
        self.person_weapon_distance_threshold = 0.15
        self.person_animal_distance_threshold = 0.30
    
    def evaluate(
        self, 
        phase1_frames: List[FramePrediction],
        phase2_frames: List[FramePrediction], 
        audio_prediction: Optional[AudioPrediction],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[List[AlertPrediction], Optional[GanPrediction]]:
        """Generate alerts based on detections"""
        
        context = context or {}
        alerts = []
        
        # Build frame lookup
        phase1_lookup = {f.frame_index: f for f in phase1_frames}
        
        # Process each Phase 2 frame
        for frame in phase2_frames:
            phase1_frame = phase1_lookup.get(frame.frame_index)
            
            # Check for armed persons
            alerts.extend(self._check_armed_person(frame, context))
            
            # Check human-animal proximity
            if phase1_frame:
                alerts.extend(self._check_human_animal_proximity(frame, phase1_frame, context))
            
            # Check night activity
            if context.get("is_night_time"):
                alerts.extend(self._check_night_activity(frame, context))
        
        # Check audio threats
        if audio_prediction and audio_prediction.label == "POACHING_THREAT":
            alerts.extend(self._check_audio_threat(audio_prediction, context))
        
        # Generate GAN prediction if alerts exist
        gan_prediction = None
        if alerts:
            gan_prediction = self._generate_gan_prediction(alerts, context)
        
        return alerts, gan_prediction
    
    def _check_armed_person(self, frame: FramePrediction, context: Dict) -> List[AlertPrediction]:
        """Check for armed persons"""
        persons = [d for d in frame.detections if d.label.lower() in PERSON_LABELS]
        weapons = [d for d in frame.detections if d.label.lower() in WEAPON_LABELS]
        
        alerts = []
        for weapon in weapons:
            evidence = AlertEvidence(
                detection_source="Phase2 Weapon Detection",
                confidence_scores={"weapon": weapon.confidence},
                threat_assessment=f"Weapon detected: {weapon.label}"
            )
            
            alerts.append(AlertPrediction(
                level="CRITICAL",
                rule="weapon_detected",
                confidence=weapon.confidence,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                evidence=evidence,
                recommended_actions=["immediate_ranger_dispatch", "alert_control_room"],
                response_time_target="< 2 minutes"
            ))
        
        return alerts
    
    def _check_human_animal_proximity(
        self, 
        phase2_frame: FramePrediction,
        phase1_frame: FramePrediction,
        context: Dict
    ) -> List[AlertPrediction]:
        """Check for humans near endangered animals"""
        persons = [d for d in phase2_frame.detections if d.label.lower() in PERSON_LABELS]
        animals = [d for d in phase1_frame.detections if d.label.lower() in self.endangered_classes]
        
        alerts = []
        for person in persons:
            for animal in animals:
                distance = self._calculate_distance(person.bbox_xyxy, animal.bbox_xyxy)
                if distance <= self.person_animal_distance_threshold:
                    evidence = AlertEvidence(
                        detection_source="Phase1 + Phase2 fusion",
                        confidence_scores={"person": person.confidence, "animal": animal.confidence},
                        threat_assessment=f"Human near {animal.label}"
                    )
                    
                    alerts.append(AlertPrediction(
                        level="CRITICAL",
                        rule="human_near_endangered_animal",
                        confidence=min(person.confidence, animal.confidence),
                        frame_index=phase2_frame.frame_index,
                        timestamp=phase2_frame.timestamp,
                        evidence=evidence,
                        recommended_actions=["immediate_investigation", "dispatch_team"],
                        response_time_target="< 90 seconds"
                    ))
        
        return alerts
    
    def _check_night_activity(self, frame: FramePrediction, context: Dict) -> List[AlertPrediction]:
        """Check for suspicious night activity"""
        persons = [d for d in frame.detections if d.label.lower() in PERSON_LABELS]
        
        if persons:
            evidence = AlertEvidence(
                detection_source="Phase2",
                confidence_scores={"person": max(p.confidence for p in persons)},
                threat_assessment="Human activity detected at night"
            )
            
            return [AlertPrediction(
                level="HIGH",
                rule="night_human_activity",
                confidence=max(p.confidence for p in persons),
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                evidence=evidence,
                recommended_actions=["investigate_immediately", "increase_patrols"],
                response_time_target="< 5 minutes"
            )]
        
        return []
    
    def _check_audio_threat(self, audio: AudioPrediction, context: Dict) -> List[AlertPrediction]:
        """Check audio for threats"""
        evidence = AlertEvidence(
            detection_source="Audio Classifier",
            confidence_scores={"audio": audio.confidence},
            threat_assessment=f"Threatening audio detected: {audio.label}"
        )
        
        return [AlertPrediction(
            level="HIGH",
            rule="poaching_audio_detected",
            confidence=audio.confidence,
            evidence=evidence,
            recommended_actions=["investigate_audio_source", "increase_surveillance"],
            response_time_target="< 3 minutes"
        )]
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate normalized distance between bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        frame_diagonal = (1920 ** 2 + 1080 ** 2) ** 0.5
        
        return distance / frame_diagonal
    
    def _generate_gan_prediction(self, alerts: List[AlertPrediction], context: Dict) -> GanPrediction:
        """Generate GAN-based threat prediction"""
        critical_count = sum(1 for a in alerts if a.level == "CRITICAL")
        high_count = sum(1 for a in alerts if a.level == "HIGH")
        
        if critical_count >= 2:
            operation_type = "professional"
            escalation = "RAPID_ESCALATION"
            risk = "CRITICAL - Immediate intervention required"
        elif critical_count >= 1 or high_count >= 2:
            operation_type = "opportunistic"
            escalation = "MODERATE_ESCALATION"
            risk = "HIGH - Enhanced monitoring needed"
        else:
            operation_type = "subsistence"
            escalation = "STABLE"
            risk = "MEDIUM - Standard monitoring"
        
        return GanPrediction(
            operation_type=operation_type,
            escalation_pattern=escalation,
            risk_assessment=risk,
            confidence=0.75
        )

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Wildlife Protection AI System",
    description="Multi-modal AI for wildlife conservation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await model_manager.load_all_models()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model_manager.is_loaded
    }

# =============================================================================
# == MODIFIED ENDPOINT ========================================================
# =============================================================================
@app.post("/api/predict/fused", response_model=AnalysisResponse)
async def predict_fused(
    video_file: UploadFile = File(...),
    context: str = Form("{}"),
):
    """
    Main prediction endpoint - processes a single video file for
    Phase 1, Phase 2, and Audio analysis.
    """
    try:
        if not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Models not yet loaded")
        
        # Parse context
        context_dict = json.loads(context)
        frame_stride = context_dict.get("frame_stride", 5)
        
        # Save uploaded video file
        # Use a unique name, preserving the extension from the original filename
        video_path = TEMP_DIR / f"{uuid.uuid4()}_{video_file.filename}"
        
        with open(video_path, "wb") as f:
            f.write(await video_file.read())
        
        # Process through models
        logger.info("Processing video through Phase 1...")
        phase1_frames = model_manager.process_video_phase1(str(video_path), frame_stride)
        
        logger.info("Processing video through Phase 2...")
        phase2_frames = model_manager.process_video_phase2(str(video_path), frame_stride)
        
        logger.info("Processing audio from video file...")
        audio_prediction = model_manager.process_audio(str(video_path))
        
        # Generate alerts
        logger.info("Generating alerts...")
        alerts, gan_prediction = model_manager.fusion_engine.evaluate(
            phase1_frames, phase2_frames, audio_prediction, context_dict
        )
        
        # Clean up temp file
        video_path.unlink()
        
        # Build response
        response = AnalysisResponse(
            phase1_frames=phase1_frames,
            phase2_frames=phase2_frames,
            audio=audio_prediction,
            alerts=alerts,
            gan_prediction=gan_prediction,
            processing_metadata={
                "frame_stride": frame_stride,
                "total_frames_processed": len(phase1_frames),
                "context": context_dict
            }
        )
        
        logger.info(f"Analysis complete - {len(alerts)} alerts generated")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
# =============================================================================
# == END OF MODIFIED ENDPOINT =================================================
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🌿 Starting Wildlife Protection AI System")
    uvicorn.run(
        "server_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )