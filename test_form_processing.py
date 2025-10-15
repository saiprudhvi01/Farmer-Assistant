#!/usr/bin/env python3
# Test script for form submission simulation

import sys
import os
sys.path.append('c:\\Users\\SAI PRUDHVI\\Downloads\\Farmer assistant')

from src.models.crop_recommendation import DiseaseDetector

def test_form_submission():
    detector = DiseaseDetector()

    # Simulate form data as it would come from the Flask request
    print("=== Testing Form Data Processing ===")

    # Test case 1: Multiple symptoms from checkboxes (simulating form.getlist('symptoms'))
    print("Test Case 1: Multiple symptoms from checkboxes")
    form_data = {
        'symptoms': ['dark spots with concentric rings', 'yellow halos around spots', 'leaf curling'],
        'custom_symptoms': '',
        'crop_type': 'tomato',
        'severity': 'medium'
    }

    # Process symptoms like the Flask app does
    symptoms = form_data.get('symptoms', [])
    custom_symptoms = form_data.get('custom_symptoms', '').strip()

    # Clean and process symptoms
    all_symptoms = []
    if symptoms:
        for symptom in symptoms:
            # Split comma-separated symptoms and clean them
            cleaned = [s.strip() for s in symptom.split(',') if s.strip()]
            all_symptoms.extend(cleaned)

    if custom_symptoms:
        # Split by comma and clean custom symptoms
        custom_cleaned = [s.strip() for s in custom_symptoms.split(',') if s.strip()]
        all_symptoms.extend(custom_cleaned)

    # Remove duplicates and empty strings
    symptoms = list(set(all_symptoms))
    symptoms = [s for s in symptoms if s]

    print(f"Processed symptoms: {symptoms}")

    # Run disease detection
    diseases = detector.detect_disease(form_data['crop_type'], symptoms=symptoms, severity=form_data['severity'])

    for disease in diseases:
        print(f"Disease: {disease['disease']}, Confidence: {disease['confidence']}%")
        if disease['confidence'] > 10:
            print(f"  Method: {disease['method']}")
            print(f"  Matched symptoms: {disease['matched_symptoms']}")
            print(f"  Reasoning: {disease['reasoning']}")
        print()

    # Test case 2: Custom symptoms only
    print("Test Case 2: Custom symptoms only")
    form_data2 = {
        'symptoms': [],
        'custom_symptoms': 'brown spots on leaves, wilting plants, yellow halos',
        'crop_type': 'tomato',
        'severity': 'high'
    }

    # Process symptoms
    symptoms = form_data2.get('symptoms', [])
    custom_symptoms = form_data2.get('custom_symptoms', '').strip()

    all_symptoms = []
    if symptoms:
        for symptom in symptoms:
            cleaned = [s.strip() for s in symptom.split(',') if s.strip()]
            all_symptoms.extend(cleaned)

    if custom_symptoms:
        custom_cleaned = [s.strip() for s in custom_symptoms.split(',') if s.strip()]
        all_symptoms.extend(custom_cleaned)

    symptoms = list(set(all_symptoms))
    symptoms = [s for s in symptoms if s]

    print(f"Processed symptoms: {symptoms}")

    diseases = detector.detect_disease(form_data2['crop_type'], symptoms=symptoms, severity=form_data2['severity'])

    for disease in diseases:
        print(f"Disease: {disease['disease']}, Confidence: {disease['confidence']}%")
        if disease['confidence'] > 10:
            print(f"  Method: {disease['method']}")
            print(f"  Matched symptoms: {disease['matched_symptoms']}")
            print(f"  Reasoning: {disease['reasoning']}")
        print()

    # Test case 3: No symptoms (should fallback)
    print("Test Case 3: No symptoms (fallback test)")
    diseases = detector.detect_disease('tomato', symptoms=[], severity='medium')

    for disease in diseases:
        print(f"Disease: {disease['disease']}, Confidence: {disease['confidence']}%")
        print()

if __name__ == "__main__":
    test_form_submission()
