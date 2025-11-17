#!/usr/bin/env python3
"""
Test script for multi-company incident classifier API
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8001/api/v1"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_list_companies():
    """Test listing all companies"""
    print_section("1. List All Companies")
    
    response = requests.get(f"{BASE_URL}/companies")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {data['count']} companies:")
        for company in data['companies']:
            status = "✓ Trained" if company['model_trained'] else "✗ Not trained"
            print(f"\n  Company: {company['company_name']} ({company['company_id']})")
            print(f"  Status: {status}")
            print(f"  Incident Types: {', '.join(company['incident_types'])}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_company_info(company_id="00110"):
    """Test getting company information"""
    print_section(f"2. Get Company Info: {company_id}")
    
    response = requests.get(f"{BASE_URL}/companies/{company_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Company: {data['company_name']}")
        print(f"  Model Trained: {'Yes ✓' if data['model_trained'] else 'No ✗'}")
        print(f"  Confidence Threshold: {data['confidence_threshold']}")
        print(f"\n  Incident Types ({len(data['incident_types'])}):")
        for i, inc_type in enumerate(data['incident_types'], 1):
            dept = data['department_mapping'].get(inc_type, 'Unknown')
            print(f"    {i}. {inc_type} → {dept}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_classify_image(company_id="00110", image_path=None):
    """Test image classification"""
    print_section(f"3. Classify Image (Company: {company_id})")
    
    if not image_path:
        print("⚠ No image provided - skipping classification test")
        print("  Usage: Provide image path as argument")
        return
    
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"  Image: {image_file.name}")
    
    with open(image_file, 'rb') as f:
        files = {'file': (image_file.name, f, 'image/jpeg')}
        response = requests.post(
            f"{BASE_URL}/classify/{company_id}",
            files=files,
            params={'top_k': 3}
        )
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"\n✓ Classification Result:")
        print(f"  Incident Type: {data['incident_type']}")
        print(f"  Confidence: {data['confidence']:.2%}")
        print(f"  Department: {data['department']}")
        print(f"\n  Top Predictions:")
        for i, pred in enumerate(data['all_predictions'], 1):
            print(f"    {i}. {pred['incident_type']} ({pred['confidence']:.2%}) → {pred['department']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_health_check():
    """Test health check endpoint"""
    print_section("0. Health Check")
    
    try:
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            print("✓ API is running")
        else:
            print(f"✗ API returned status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API")
        print("  Make sure the server is running:")
        print("  uvicorn app.main:app --reload --port 8001")
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n🔬 Testing Multi-Company Incident Classifier API")
    print("="*60)
    
    # Check if API is running
    if not test_health_check():
        return
    
    # Test listing companies
    test_list_companies()
    
    # Test getting company info
    test_company_info("00110")
    
    # Test classification (provide image path to test)
    # test_classify_image("00110", "path/to/test/image.jpg")
    test_classify_image("00110")
    
    print("\n" + "="*60)
    print("✓ Testing Complete!")
    print("="*60)
    print("\nTo test image classification:")
    print("  1. Add training images to companies/00110/training_data/train/")
    print("  2. Train model: python scripts/train_company_model.py 00110")
    print("  3. Test with image: Uncomment line in this script\n")


if __name__ == "__main__":
    main()
