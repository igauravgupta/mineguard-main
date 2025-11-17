"""
Incident Classifier Model - Multi-Company Support
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json
from app.core.config import settings

logger = logging.getLogger(__name__)


class IncidentClassifier(nn.Module):
    """CNN-based incident classifier using EfficientNet-B0"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(IncidentClassifier, self).__init__()
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class CompanyModelManager:
    """Manages company-specific models and configurations"""
    
    def __init__(self):
        self.device = torch.device("cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.loaded_models: Dict[str, Dict] = {}  # Cache for loaded models
        self.companies_dir = Path(settings.BASE_DIR) / "companies"
    
    def get_company_config(self, company_id: str) -> Dict:
        """Load company configuration"""
        config_path = self.companies_dir / company_id / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Company {company_id} configuration not found")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_company_model(self, company_id: str) -> Dict:
        """Load company-specific model"""
        # Check cache first
        if company_id in self.loaded_models:
            logger.info(f"Using cached model for company {company_id}")
            return self.loaded_models[company_id]
        
        try:
            # Load company config
            config = self.get_company_config(company_id)
            incident_types = config['incident_types']
            
            # Initialize model
            model = IncidentClassifier(
                num_classes=len(incident_types),
                pretrained=True
            )
            
            # Load trained weights
            model_path = self.companies_dir / company_id / "models" / "best_model.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded trained model for company {company_id}")
            else:
                logger.warning(
                    f"No trained model found for company {company_id} at {model_path}. "
                    "Using pretrained backbone only."
                )
            
            model = model.to(self.device)
            model.eval()
            
            # Cache the loaded model
            model_data = {
                'model': model,
                'classes': incident_types,
                'config': config
            }
            self.loaded_models[company_id] = model_data
            
            logger.info(f"Model loaded successfully for company {company_id}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model for company {company_id}: {e}")
            raise
    
    def predict(self, company_id: str, image_tensor: torch.Tensor, top_k: int = 3) -> Dict:
        """
        Predict incident type from image tensor for a specific company
        
        Args:
            company_id: Company identifier
            image_tensor: Preprocessed image tensor
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and routing information
        """
        try:
            # Load company model
            model_data = self.load_company_model(company_id)
            model = model_data['model']
            classes = model_data['classes']
            config = model_data['config']
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(
                probabilities[0],
                min(top_k, len(classes))
            )
            
            # Format results
            all_predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                incident_type = classes[int(idx)]
                all_predictions.append({
                    "incident_type": incident_type,
                    "confidence": float(prob),
                    "department": config['department_mapping'].get(incident_type, "Unknown")
                })
            
            return {
                "company_id": company_id,
                "company_name": config['company_name'],
                "incident_type": classes[int(top_indices[0])],
                "confidence": float(top_probs[0]),
                "department": config['department_mapping'].get(classes[int(top_indices[0])], "Unknown"),
                "all_predictions": all_predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction error for company {company_id}: {e}")
            raise
    
    def list_companies(self) -> List[Dict]:
        """List all available companies"""
        companies = []
        if not self.companies_dir.exists():
            return companies
        
        for company_path in self.companies_dir.iterdir():
            if company_path.is_dir():
                config_path = company_path / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        model_path = company_path / "models" / "best_model.pth"
                        companies.append({
                            "company_id": config['company_id'],
                            "company_name": config['company_name'],
                            "incident_types": config['incident_types'],
                            "departments": config['departments'],
                            "model_trained": model_path.exists()
                        })
                    except Exception as e:
                        logger.warning(f"Error loading config for {company_path.name}: {e}")
        
        return companies
    
    def clear_cache(self, company_id: Optional[str] = None):
        """Clear cached models"""
        if company_id:
            if company_id in self.loaded_models:
                del self.loaded_models[company_id]
                logger.info(f"Cleared cache for company {company_id}")
        else:
            self.loaded_models.clear()
            logger.info("Cleared all cached models")


# Global model manager instance
model_manager = CompanyModelManager()

