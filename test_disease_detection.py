#!/usr/bin/env python3
# Test script for disease detection

import sys
import os
sys.path.append('c:\\Users\\SAI PRUDHVI\\Downloads\\Farmer assistant')

from src.models.crop_recommendation import DiseaseDetector

def test_disease_detection():
    detector = DiseaseDetector()

    # Test case 1: Tomato with early blight symptoms
    print("=== Test Case 1: Tomato Early Blight ===")
    symptoms = ["dark spots with concentric rings", "yellow halos around spots"]
    diseases = detector.detect_disease("tomato", symptoms=symptoms, severity="medium")
    print(f"Symptoms: {symptoms}")
    for disease in diseases:
        print(f"Disease: {disease['disease']}, Confidence: {disease['confidence']}%")
        print(f"Matched symptoms: {disease['matched_symptoms']}")
        print(f"Reasoning: {disease['reasoning']}")
        print()

    # Test case 2: Tomato with no symptoms (should fallback)
    print("=== Test Case 2: Tomato No Symptoms ===")
    diseases = detector.detect_disease("tomato", symptoms=[], severity="medium")
    print("Symptoms: []")
    for disease in diseases:
        print(f"Disease: {disease['disease']}, Confidence: {disease['confidence']}%")
        print()

    # Test case 3: Generic symptoms that should match multiple diseases
    print("=== Test Case 3: Generic Symptoms ===")
    symptoms = ["yellowing leaves", "spots on leaves"]
    diseases = detector.detect_disease("tomato", symptoms=symptoms, severity="medium")
    print(f"Symptoms: {symptoms}")
    for disease in diseases:
        print(f"Disease: {disease['disease']}, Confidence: {disease['confidence']}%")
        print()

if __name__ == "__main__":
    test_disease_detection()
