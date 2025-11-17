# Incident Classifier API

AI-powered incident image classification with multi-company support. Each company gets custom incident types, departments, and trained models.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload --port 8001

# Test
curl http://localhost:8001/api/v1/companies
```

## API Endpoints

**List Companies:** `GET /api/v1/companies`

**Company Info:** `GET /api/v1/companies/{company_id}`

**Classify Image:** `POST /api/v1/classify/{company_id}`

```bash
curl -X POST "http://localhost:8001/api/v1/classify/00110" -F "file=@image.jpg"
```

## Add New Company

1. Create structure:

```bash
mkdir -p companies/NEW_ID/{training_data/train,models}
```

2. Create `companies/NEW_ID/config.json`:

```json
{
  "company_id": "NEW_ID",
  "company_name": "Company Name",
  "departments": ["Dept1", "Dept2"],
  "incident_types": ["Type1", "Type2"],
  "department_mapping": { "Type1": "Dept1", "Type2": "Dept2" },
  "confidence_threshold": 0.6
}
```

3. Add images to `companies/NEW_ID/training_data/train/Type1/`, `Type2/`, etc. (100-200+ per type)

4. Train:

```bash
python scripts/train_company_model.py NEW_ID --epochs 30
```

## Sample Companies

**00110** - Sample Mining Company (5 incident types)
**00220** - Industrial Mining Corp (4 incident types)

## Tech Stack

- FastAPI + PyTorch
- EfficientNet-B0 (transfer learning)
- Per-company model isolation
