"""
API Routes - Multi-Company Support
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Path
from typing import Optional, List
import logging
from app.models.classifier import model_manager
from app.utils.image_processing import process_image, validate_image
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/companies")
async def list_companies():
    """Get list of all registered companies"""
    try:
        companies = model_manager.list_companies()
        return {
            "success": True,
            "companies": companies,
            "count": len(companies)
        }
    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        raise HTTPException(status_code=500, detail="Failed to list companies")


@router.get("/companies/{company_id}")
async def get_company_info(company_id: str = Path(..., description="Company ID")):
    """Get company configuration and incident types"""
    try:
        config = model_manager.get_company_config(company_id)
        model_path = model_manager.companies_dir / company_id / "models" / "best_model.pth"
        
        return {
            "success": True,
            "company_id": config['company_id'],
            "company_name": config['company_name'],
            "incident_types": config['incident_types'],
            "departments": config['departments'],
            "department_mapping": config['department_mapping'],
            "model_trained": model_path.exists(),
            "confidence_threshold": config.get('confidence_threshold', 0.6)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Company {company_id} not found")
    except Exception as e:
        logger.error(f"Error getting company info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get company info")


@router.post("/classify/{company_id}")
async def classify_image(
    company_id: str = Path(..., description="Company ID"),
    file: UploadFile = File(...),
    top_k: int = 3
):
    """
    Classify incident type from uploaded image for specific company
    
    - **company_id**: Company identifier (e.g., 00110)
    - **file**: Image file (jpg, png, etc.)
    - **top_k**: Number of top predictions to return (default: 3)
    """
    try:
        # Read image
        image_bytes = await file.read()
        
        # Validate image
        validate_image(image_bytes, settings.MAX_IMAGE_SIZE)
        
        # Process image
        image_tensor = process_image(image_bytes)
        
        # Get prediction
        result = model_manager.predict(company_id, image_tensor, top_k)
        
        return {
            "success": True,
            "data": result
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Company {company_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")


@router.post("/classify-with-verification/{company_id}")
async def classify_with_verification(
    company_id: str = Path(..., description="Company ID"),
    file: UploadFile = File(...),
    user_description: Optional[str] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    top_k: int = Form(3)
):
    """
    Classify incident and verify with user description for specific company
    
    - **company_id**: Company identifier
    - **file**: Image file
    - **user_description**: User's text description of incident (optional)
    - **confidence_threshold**: Minimum confidence for auto-accept (default from company config)
    - **top_k**: Number of top predictions (default: 3)
    
    Returns:
    - Predicted incident type
    - Confidence score
    - Whether it needs manual review
    - Responsible department
    - Recommendation
    """
    try:
        # Get company config
        config = model_manager.get_company_config(company_id)
        
        # Use company's threshold if not provided
        threshold = confidence_threshold or config.get('confidence_threshold', 0.6)
        
        # Read and validate image
        image_bytes = await file.read()
        validate_image(image_bytes, settings.MAX_IMAGE_SIZE)
        
        # Process image
        image_tensor = process_image(image_bytes)
        
        # Get prediction
        prediction = model_manager.predict(company_id, image_tensor, top_k)
        
        # Determine if manual review needed
        needs_review = prediction["confidence"] < threshold
        
        return {
            "success": True,
            "data": {
                "company_id": company_id,
                "company_name": prediction["company_name"],
                "predicted_incident_type": prediction["incident_type"],
                "confidence": prediction["confidence"],
                "responsible_department": prediction["department"],
                "needs_manual_review": needs_review,
                "recommendation": "Auto-approved for reporting" if not needs_review else "Manual review recommended",
                "all_predictions": prediction["all_predictions"],
                "user_description": user_description,
                "threshold_used": threshold
            }
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Company {company_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

