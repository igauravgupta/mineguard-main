# How to Add Training Data and Train Model

## Step 1: Organize Training Images

Add images to the company's training folder with this structure:

```
companies/00110/training_data/train/
├── Equipment Failure/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ... (100-200+ images)
├── Safety Hazard/
│   └── ... (100-200+ images)
├── Environmental Issue/
│   └── ... (100-200+ images)
├── Structural Damage/
│   └── ... (100-200+ images)
└── Fire or Explosion/
    └── ... (100-200+ images)
```

**Important:**

- Folder names must EXACTLY match incident types in `config.json`
- 100-200+ images per incident type recommended
- Supported formats: JPG, PNG
- Clear, well-lit images work best

## Step 2: Train the Model

```bash
# For company 00110
python scripts/train_company_model.py 00110 --epochs 30

# For company 00220
python scripts/train_company_model.py 00220 --epochs 30
```

**Training Options:**

```bash
--epochs 30          # Number of training cycles (20-50 recommended)
--batch-size 32      # Reduce if out of memory (16, 8)
--learning-rate 0.001  # Learning rate
```

## Step 3: Wait for Training

Training time depends on:

- **GPU**: 10-30 minutes
- **CPU**: 1-3 hours

You'll see progress like:

```
Epoch 1/30
Training: 100% loss: 0.5234 acc: 75.23%
Validation: 100%
Train Loss: 0.5234 | Train Acc: 75.23%
Val Loss: 0.4123 | Val Acc: 80.45%
✓ Saved best model (Val Acc: 80.45%)
```

## Step 4: Model Saved

After training, model is saved to:

```
companies/00110/models/best_model.pth
```

## Step 5: Use the API

```bash
# Start API
uvicorn app.main:app --reload --port 8001

# Test classification
curl -X POST "http://localhost:8001/api/v1/classify/00110" \
  -F "file=@test_image.jpg"
```

## Expected Accuracy

| Total Images | Images per Type | Expected Accuracy |
| ------------ | --------------- | ----------------- |
| 250-500      | 50-100          | 65-75%            |
| 500-1000     | 100-200         | 75-85%            |
| 1000-2500    | 200-500         | 85-95%            |

## Tips for Better Results

1. **More data is better** - Aim for 200+ images per type
2. **Quality matters** - Clear, focused images
3. **Variety helps** - Different angles, lighting, conditions
4. **Balance dataset** - Similar number per incident type
5. **Correct labels** - Ensure images are in correct folders
