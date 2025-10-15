# app/models/crop_recommendation.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import requests
from datetime import datetime
import os

class CropRecommender:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_data = {
            'rice': {'N': 80, 'P': 40, 'K': 40, 'temperature': 25, 'humidity': 80, 'ph': 6.5, 'rainfall': 150, 'season': 'Kharif'},
            'wheat': {'N': 70, 'P': 30, 'K': 30, 'temperature': 20, 'humidity': 60, 'ph': 7.0, 'rainfall': 100, 'season': 'Rabi'},
            'maize': {'N': 90, 'P': 40, 'K': 35, 'temperature': 27, 'humidity': 70, 'ph': 6.0, 'rainfall': 90, 'season': 'Kharif'},
            'cotton': {'N': 60, 'P': 25, 'K': 25, 'temperature': 30, 'humidity': 65, 'ph': 7.5, 'rainfall': 80, 'season': 'Kharif'},
            'sugarcane': {'N': 120, 'P': 50, 'K': 60, 'temperature': 28, 'humidity': 75, 'ph': 7.0, 'rainfall': 120, 'season': 'Kharif'},
            'tomato': {'N': 100, 'P': 45, 'K': 50, 'temperature': 24, 'humidity': 70, 'ph': 6.5, 'rainfall': 60, 'season': 'Rabi'},
            'potato': {'N': 90, 'P': 35, 'K': 40, 'temperature': 18, 'humidity': 65, 'ph': 6.0, 'rainfall': 80, 'season': 'Rabi'},
            'onion': {'N': 80, 'P': 30, 'K': 35, 'temperature': 22, 'humidity': 60, 'ph': 6.5, 'rainfall': 70, 'season': 'Rabi'},
            'soybean': {'N': 40, 'P': 20, 'K': 20, 'temperature': 26, 'humidity': 70, 'ph': 6.5, 'rainfall': 100, 'season': 'Kharif'},
            'groundnut': {'N': 30, 'P': 15, 'K': 15, 'temperature': 28, 'humidity': 65, 'ph': 6.8, 'rainfall': 80, 'season': 'Kharif'}
        }
        self.load_or_train_model()

    def load_or_train_model(self):
        try:
            import joblib
            self.model = joblib.load('models/crop_recommendation_model.pkl')
            self.label_encoder = joblib.load('models/crop_label_encoder.pkl')
        except:
            self.train_model()

    def train_model(self):
        try:
            # Generate synthetic data based on ideal conditions
            data = []
            for crop, params in self.crop_data.items():
                for _ in range(100):
                    row = {}
                    for k, v in params.items():
                        if k == 'season':
                            row[k] = v  # Keep season as string
                        else:
                            row[k] = np.random.normal(v, v*0.1)  # Apply noise only to numeric values
                    row['crop'] = crop
                    data.append(row)

            df = pd.DataFrame(data)
            X = df[self.feature_columns]
            y = df['crop']

            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y_encoded)

            # Save the model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/crop_recommendation_model.pkl')
            joblib.dump(self.label_encoder, 'models/crop_label_encoder.pkl')
        except ImportError:
            # sklearn not available, model will remain None and we'll use rule-based
            print("Warning: sklearn not available, using rule-based crop recommendation only")
            self.model = None
            self.label_encoder = None

    def get_weather_data(self, location):
        """Get weather data from NASA POWER API"""
        try:
            # NASA POWER API endpoint for location data
            base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                'parameters': 'T2M,PRECTOT,T2M_MAX,T2M_MIN,RH2M,TS',
                'community': 'RE',
                'longitude': '-0.1257',  # Default to London for demo
                'latitude': '51.5085',
                'start': '2020',
                'end': '2025',
                'format': 'JSON'
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['properties']['parameter']['T2M']['2025'][-1] - 273.15,  # Convert from Kelvin to Celsius
                    'humidity': data['properties']['parameter']['RH2M']['2025'][-1],
                    'rainfall': data['properties']['parameter']['PRECTOT']['2025'][-1] * 30  # Convert to monthly mm
                }
        except:
            pass

        # Return default values if API fails
        return {
            'temperature': 25,
            'humidity': 70,
            'rainfall': 100
        }

    def rule_based_recommendation(self, soil_data, location):
        """Rule-based crop recommendation based on conditions"""
        recommendations = []

        for crop, params in self.crop_data.items():
            score = 0

            # Check temperature suitability
            temp_diff = abs(soil_data.get('temperature', 25) - params['temperature'])
            if temp_diff <= 5:
                score += 3
            elif temp_diff <= 10:
                score += 2
            elif temp_diff <= 15:
                score += 1

            # Check pH suitability
            ph = soil_data.get('ph', 7.0)
            if params['ph'] - 0.5 <= ph <= params['ph'] + 0.5:
                score += 3
            elif params['ph'] - 1 <= ph <= params['ph'] + 1:
                score += 2
            elif params['ph'] - 1.5 <= ph <= params['ph'] + 1.5:
                score += 1

            # Check rainfall suitability
            rainfall = soil_data.get('rainfall', 100)
            rainfall_diff = abs(rainfall - params['rainfall'])
            if rainfall_diff <= 20:
                score += 3
            elif rainfall_diff <= 40:
                score += 2
            elif rainfall_diff <= 60:
                score += 1

            if score >= 6:
                recommendations.append({
                    'crop': crop,
                    'score': score,
                    'season': params['season'],
                    'confidence': min(score * 15, 95)  # Convert score to percentage
                })

        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]

    def recommend_crops(self, soil_data, location):
        """Main method to recommend crops"""
        # Get weather data if location is provided
        if location and location.strip():
            weather_data = self.get_weather_data(location)
            soil_data.update(weather_data)

        # Try ML model first
        if self.model and self.label_encoder:
            try:
                features = np.array([[soil_data.get(col, 0) for col in self.feature_columns]])
                prediction = self.model.predict_proba(features)
                top_indices = prediction[0].argsort()[-3:][::-1]

                ml_recommendations = []
                for idx in top_indices:
                    crop = self.label_encoder.inverse_transform([idx])[0]
                    confidence = prediction[0][idx] * 100
                    if confidence > 10:  # Only include if confidence > 10%
                        ml_recommendations.append({
                            'crop': crop,
                            'confidence': round(confidence, 1),
                            'method': 'ML Model',
                            'season': self.crop_data[crop]['season']
                        })

                if ml_recommendations:
                    return ml_recommendations
            except Exception as e:
                print(f"ML model failed: {e}")

class DiseaseDetector:
    def __init__(self):
        # Rule-based disease database for selected crops
        self.disease_rules = [
            # Rice
            {
                "crop": "Rice",
                "symptoms": ["yellow leaves", "brown spots", "stunted growth", "leaf rot", "whitish powder"],
                "disease": "Brown Spot",
                "pesticide": "Mancozeb",
                "cost_per_ha": 100
            },
            {
                "crop": "Rice",
                "symptoms": ["leaf blight", "yellow stripes", "necrotic lesions", "wilt", "dry tips"],
                "disease": "Bacterial Leaf Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 120
            },
            # Wheat
            {
                "crop": "Wheat",
                "symptoms": ["yellow streaks", "leaf rust", "red pustules", "brown spots", "premature leaf death"],
                "disease": "Leaf Rust",
                "pesticide": "Propiconazole",
                "cost_per_ha": 90
            },
            {
                "crop": "Wheat",
                "symptoms": ["white powdery leaves", "stunted growth", "deformed spikes", "yellow leaves", "reduced yield"],
                "disease": "Powdery Mildew",
                "pesticide": "Sulfur",
                "cost_per_ha": 80
            },
            # Maize
            {
                "crop": "Maize",
                "symptoms": ["yellow leaves", "cob rot", "brown lesions", "brown spots", "stunted growth", "wilting"],
                "disease": "Maize Streak Virus",
                "pesticide": "Imidacloprid",
                "cost_per_ha": 110
            },
            # Cotton
            {
                "crop": "Cotton",
                "symptoms": ["leaf spots", "yellowing", "boll rot", "wilt", "defoliation"],
                "disease": "Bacterial Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 130
            },
            # Sugarcane
            {
                "crop": "Sugarcane",
                "symptoms": ["reddish streaks", "wilting", "brown spots", "stunted growth", "leaf curling"],
                "disease": "Red Rot",
                "pesticide": "Carbendazim",
                "cost_per_ha": 150
            },
            # Tomato
            {
                "crop": "Tomato",
                "symptoms": ["yellow leaves", "spots on leaves", "wilting", "fruit rot", "leaf curl"],
                "disease": "Early Blight",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 120
            },
            # Potato
            {
                "crop": "Potato",
                "symptoms": ["brown spots", "leaf curl", "wilting", "tuber rot", "yellow leaves"],
                "disease": "Late Blight",
                "pesticide": "Metalaxyl",
                "cost_per_ha": 110
            },
            # Onion
            {
                "crop": "Onion",
                "symptoms": ["yellow leaves", "neck rot", "leaf spots", "wilting", "soft rot"],
                "disease": "Purple Blotch",
                "pesticide": "Mancozeb",
                "cost_per_ha": 95
            },
            # Soybean
            {
                "crop": "Soybean",
                "symptoms": ["yellow leaves", "leaf spots", "wilt", "stem lesions", "defoliation"],
                "disease": "Soybean Rust",
                "pesticide": "Trifloxystrobin",
                "cost_per_ha": 100
            },
            # Groundnut
            {
                "crop": "Groundnut",
                "symptoms": ["yellow leaves", "leaf spots", "wilting", "pod rot", "stunted growth"],
                "disease": "Leaf Spot",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 85
            }
        ]

    def detect_disease(self, crop, symptoms):
        """Detect diseases based on symptoms for a given crop"""
        detected = []
        for rule in self.disease_rules:
            if rule["crop"].lower() == crop.lower():
                rule_symptoms = [s.strip().lower() for s in rule["symptoms"]]
                matched = []
                for s in symptoms:
                    for rs in rule_symptoms:
                        if set(s.split()) & set(rs.split()):  # Check for common words
                            matched.append(s)
                            break
                if len(matched) > 0:
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": matched
                    })
        # Fallback: if no matches, return the first disease for the crop
        if not detected:
            for rule in self.disease_rules:
                if rule["crop"].lower() == crop.lower():
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": []  # No matched symptoms
                    })
                    break
        return detected

    def causal_ai_analysis(self, crop, results):
        """Generate detailed causal analysis for detected diseases using disease_data"""
        analysis = []
        for r in results:
            pesticide = r.get('pesticide', 'Consult expert for appropriate pesticide')
            cost = r.get('cost_per_ha', 0)
            disease = r['disease']
            
            # Get detailed disease info if available
            disease_info = self.disease_data.get(disease.lower().replace(' ', '_'), {})
            
            para = f"""
Causal Analysis for {crop}:

The disease '{disease}' was detected primarily due to the presence of symptoms {r['matched_symptoms']}. 
These symptoms indicate stress or infection in the crop that, if left unchecked, could lead to significant yield loss, stunted growth, or complete crop failure. 

Immediate application of the recommended pesticide '{pesticide}' at a cost of ${cost} per hectare is advised to mitigate the spread. 
Monitoring and preventive measures should follow to avoid recurrence, ensuring crop health and productivity.

"""
            if disease_info:
                para += f"""
Additional Insights:
- **Type**: {disease_info.get('type', 'Unknown')}
- **Causes**: {', '.join(disease_info.get('causes', []))}
- **Effects**: {disease_info.get('effects', 'Reduced crop health and productivity')}
- **Prevention**: {', '.join(disease_info.get('prevention', []))}
- **Spread Rate**: {disease_info.get('spread_rate', 'Unknown')}
- **Economic Impact**: {disease_info.get('economic_impact', 'Variable')}
- **Treatment Cost**: {disease_info.get('treatment_cost', 'Variable')}
"""
            analysis.append(para.strip())
        return analysis
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import requests
from datetime import datetime
import os

class CropRecommender:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_data = {
            'rice': {'N': 80, 'P': 40, 'K': 40, 'temperature': 25, 'humidity': 80, 'ph': 6.5, 'rainfall': 150, 'season': 'Kharif'},
            'wheat': {'N': 70, 'P': 30, 'K': 30, 'temperature': 20, 'humidity': 60, 'ph': 7.0, 'rainfall': 100, 'season': 'Rabi'},
            'maize': {'N': 90, 'P': 40, 'K': 35, 'temperature': 27, 'humidity': 70, 'ph': 6.0, 'rainfall': 90, 'season': 'Kharif'},
            'cotton': {'N': 60, 'P': 25, 'K': 25, 'temperature': 30, 'humidity': 65, 'ph': 7.5, 'rainfall': 80, 'season': 'Kharif'},
            'sugarcane': {'N': 120, 'P': 50, 'K': 60, 'temperature': 28, 'humidity': 75, 'ph': 7.0, 'rainfall': 120, 'season': 'Kharif'},
            'tomato': {'N': 100, 'P': 45, 'K': 50, 'temperature': 24, 'humidity': 70, 'ph': 6.5, 'rainfall': 60, 'season': 'Rabi'},
            'potato': {'N': 90, 'P': 35, 'K': 40, 'temperature': 18, 'humidity': 65, 'ph': 6.0, 'rainfall': 80, 'season': 'Rabi'},
            'onion': {'N': 80, 'P': 30, 'K': 35, 'temperature': 22, 'humidity': 60, 'ph': 6.5, 'rainfall': 70, 'season': 'Rabi'},
            'soybean': {'N': 40, 'P': 20, 'K': 20, 'temperature': 26, 'humidity': 70, 'ph': 6.5, 'rainfall': 100, 'season': 'Kharif'},
            'groundnut': {'N': 30, 'P': 15, 'K': 15, 'temperature': 28, 'humidity': 65, 'ph': 6.8, 'rainfall': 80, 'season': 'Kharif'}
        }
        self.load_or_train_model()

    def load_or_train_model(self):
        try:
            import joblib
            self.model = joblib.load('models/crop_recommendation_model.pkl')
            self.label_encoder = joblib.load('models/crop_label_encoder.pkl')
        except:
            self.train_model()

    def train_model(self):
        try:
            # Generate synthetic data based on ideal conditions
            data = []
            for crop, params in self.crop_data.items():
                for _ in range(100):
                    row = {}
                    for k, v in params.items():
                        if k == 'season':
                            row[k] = v  # Keep season as string
                        else:
                            row[k] = np.random.normal(v, v*0.1)  # Apply noise only to numeric values
                    row['crop'] = crop
                    data.append(row)

            df = pd.DataFrame(data)
            X = df[self.feature_columns]
            y = df['crop']

            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y_encoded)

            # Save the model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/crop_recommendation_model.pkl')
            joblib.dump(self.label_encoder, 'models/crop_label_encoder.pkl')
        except ImportError:
            # sklearn not available, model will remain None and we'll use rule-based
            print("Warning: sklearn not available, using rule-based crop recommendation only")
            self.model = None
            self.label_encoder = None

    def get_weather_data(self, location):
        """Get weather data from NASA POWER API"""
        try:
            # NASA POWER API endpoint for location data
            base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                'parameters': 'T2M,PRECTOT,T2M_MAX,T2M_MIN,RH2M,TS',
                'community': 'RE',
                'longitude': '-0.1257',  # Default to London for demo
                'latitude': '51.5085',
                'start': '2020',
                'end': '2025',
                'format': 'JSON'
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['properties']['parameter']['T2M']['2025'][-1] - 273.15,  # Convert from Kelvin to Celsius
                    'humidity': data['properties']['parameter']['RH2M']['2025'][-1],
                    'rainfall': data['properties']['parameter']['PRECTOT']['2025'][-1] * 30  # Convert to monthly mm
                }
        except:
            pass

        # Return default values if API fails
        return {
            'temperature': 25,
            'humidity': 70,
            'rainfall': 100
        }

    def rule_based_recommendation(self, soil_data, location):
        """Rule-based crop recommendation based on conditions"""
        recommendations = []

        for crop, params in self.crop_data.items():
            score = 0

            # Check temperature suitability
            temp_diff = abs(soil_data.get('temperature', 25) - params['temperature'])
            if temp_diff <= 5:
                score += 3
            elif temp_diff <= 10:
                score += 2
            elif temp_diff <= 15:
                score += 1

            # Check pH suitability
            ph = soil_data.get('ph', 7.0)
            if params['ph'] - 0.5 <= ph <= params['ph'] + 0.5:
                score += 3
            elif params['ph'] - 1 <= ph <= params['ph'] + 1:
                score += 2
            elif params['ph'] - 1.5 <= ph <= params['ph'] + 1.5:
                score += 1

            # Check rainfall suitability
            rainfall = soil_data.get('rainfall', 100)
            rainfall_diff = abs(rainfall - params['rainfall'])
            if rainfall_diff <= 20:
                score += 3
            elif rainfall_diff <= 40:
                score += 2
            elif rainfall_diff <= 60:
                score += 1

            if score >= 6:
                recommendations.append({
                    'crop': crop,
                    'score': score,
                    'season': params['season'],
                    'confidence': min(score * 15, 95)  # Convert score to percentage
                })

        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]

    def recommend_crops(self, soil_data, location):
        """Main method to recommend crops"""
        # Get weather data if location is provided
        if location and location.strip():
            weather_data = self.get_weather_data(location)
            soil_data.update(weather_data)

        # Try ML model first
        if self.model and self.label_encoder:
            try:
                features = np.array([[soil_data.get(col, 0) for col in self.feature_columns]])
                prediction = self.model.predict_proba(features)
                top_indices = prediction[0].argsort()[-3:][::-1]

                ml_recommendations = []
                for idx in top_indices:
                    crop = self.label_encoder.inverse_transform([idx])[0]
                    confidence = prediction[0][idx] * 100
                    if confidence > 10:  # Only include if confidence > 10%
                        ml_recommendations.append({
                            'crop': crop,
                            'confidence': round(confidence, 1),
                            'method': 'ML Model',
                            'season': self.crop_data[crop]['season']
                        })

                if ml_recommendations:
                    return ml_recommendations
            except Exception as e:
                print(f"ML model failed: {e}")

class DiseaseDetector:
    def __init__(self):
        # Rule-based disease database for selected crops
        self.disease_rules = [
            # Rice
            {
                "crop": "Rice",
                "symptoms": ["yellow leaves", "brown spots", "stunted growth", "leaf rot", "whitish powder"],
                "disease": "Brown Spot",
                "pesticide": "Mancozeb",
                "cost_per_ha": 100
            },
            {
                "crop": "Rice",
                "symptoms": ["leaf blight", "yellow stripes", "necrotic lesions", "wilt", "dry tips"],
                "disease": "Bacterial Leaf Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 120
            },
            # Wheat
            {
                "crop": "Wheat",
                "symptoms": ["yellow streaks", "leaf rust", "red pustules", "brown spots", "premature leaf death"],
                "disease": "Leaf Rust",
                "pesticide": "Propiconazole",
                "cost_per_ha": 90
            },
            {
                "crop": "Wheat",
                "symptoms": ["white powdery leaves", "stunted growth", "deformed spikes", "yellow leaves", "reduced yield"],
                "disease": "Powdery Mildew",
                "pesticide": "Sulfur",
                "cost_per_ha": 80
            },
            # Maize
            {
                "crop": "Maize",
                "symptoms": ["yellow leaves", "cob rot", "brown lesions", "brown spots", "stunted growth", "wilting"],
                "disease": "Maize Streak Virus",
                "pesticide": "Imidacloprid",
                "cost_per_ha": 110
            },
            # Cotton
            {
                "crop": "Cotton",
                "symptoms": ["leaf spots", "yellowing", "boll rot", "wilt", "defoliation"],
                "disease": "Bacterial Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 130
            },
            # Sugarcane
            {
                "crop": "Sugarcane",
                "symptoms": ["reddish streaks", "wilting", "brown spots", "stunted growth", "leaf curling"],
                "disease": "Red Rot",
                "pesticide": "Carbendazim",
                "cost_per_ha": 150
            },
            # Tomato
            {
                "crop": "Tomato",
                "symptoms": ["yellow leaves", "spots on leaves", "wilting", "fruit rot", "leaf curl"],
                "disease": "Early Blight",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 120
            },
            # Potato
            {
                "crop": "Potato",
                "symptoms": ["brown spots", "leaf curl", "wilting", "tuber rot", "yellow leaves"],
                "disease": "Late Blight",
                "pesticide": "Metalaxyl",
                "cost_per_ha": 110
            },
            # Onion
            {
                "crop": "Onion",
                "symptoms": ["yellow leaves", "neck rot", "leaf spots", "wilting", "soft rot"],
                "disease": "Purple Blotch",
                "pesticide": "Mancozeb",
                "cost_per_ha": 95
            },
            # Soybean
            {
                "crop": "Soybean",
                "symptoms": ["yellow leaves", "leaf spots", "wilt", "stem lesions", "defoliation"],
                "disease": "Soybean Rust",
                "pesticide": "Trifloxystrobin",
                "cost_per_ha": 100
            },
            # Groundnut
            {
                "crop": "Groundnut",
                "symptoms": ["yellow leaves", "leaf spots", "wilting", "pod rot", "stunted growth"],
                "disease": "Leaf Spot",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 85
            }
        ]

    def detect_disease(self, crop, symptoms):
        """Detect diseases based on symptoms for a given crop"""
        detected = []
        for rule in self.disease_rules:
            if rule["crop"].lower() == crop.lower():
                rule_symptoms = [s.strip().lower() for s in rule["symptoms"]]
                matched = []
                for s in symptoms:
                    for rs in rule_symptoms:
                        if set(s.split()) & set(rs.split()):  # Check for common words
                            matched.append(s)
                            break
                if len(matched) > 0:
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": matched
                    })
        # Fallback: if no matches, return the first disease for the crop
        if not detected:
            for rule in self.disease_rules:
                if rule["crop"].lower() == crop.lower():
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": []  # No matched symptoms
                    })
                    break
        return detected

    def causal_ai_analysis(self, crop, results):
        """Generate detailed causal analysis for detected diseases using disease_data"""
        analysis = []
        for r in results:
            pesticide = r.get('pesticide', 'Consult expert for appropriate pesticide')
            cost = r.get('cost_per_ha', 0)
            disease = r['disease']
            
            # Get detailed disease info if available
            disease_info = self.disease_data.get(disease.lower().replace(' ', '_'), {})
            
            para = f"""
Causal Analysis for {crop}:

The disease '{disease}' was detected primarily due to the presence of symptoms {r['matched_symptoms']}. 
These symptoms indicate stress or infection in the crop that, if left unchecked, could lead to significant yield loss, stunted growth, or complete crop failure. 

Immediate application of the recommended pesticide '{pesticide}' at a cost of ${cost} per hectare is advised to mitigate the spread. 
Monitoring and preventive measures should follow to avoid recurrence, ensuring crop health and productivity.

"""
            if disease_info:
                para += f"""
Additional Insights:
- **Type**: {disease_info.get('type', 'Unknown')}
- **Causes**: {', '.join(disease_info.get('causes', []))}
- **Effects**: {disease_info.get('effects', 'Reduced crop health and productivity')}
- **Prevention**: {', '.join(disease_info.get('prevention', []))}
- **Spread Rate**: {disease_info.get('spread_rate', 'Unknown')}
- **Economic Impact**: {disease_info.get('economic_impact', 'Variable')}
- **Treatment Cost**: {disease_info.get('treatment_cost', 'Variable')}
"""
            analysis.append(para.strip())
        return analysis
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import requests
from datetime import datetime
import os

class CropRecommender:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_data = {
            'rice': {'N': 80, 'P': 40, 'K': 40, 'temperature': 25, 'humidity': 80, 'ph': 6.5, 'rainfall': 150, 'season': 'Kharif'},
            'wheat': {'N': 70, 'P': 30, 'K': 30, 'temperature': 20, 'humidity': 60, 'ph': 7.0, 'rainfall': 100, 'season': 'Rabi'},
            'maize': {'N': 90, 'P': 40, 'K': 35, 'temperature': 27, 'humidity': 70, 'ph': 6.0, 'rainfall': 90, 'season': 'Kharif'},
            'cotton': {'N': 60, 'P': 25, 'K': 25, 'temperature': 30, 'humidity': 65, 'ph': 7.5, 'rainfall': 80, 'season': 'Kharif'},
            'sugarcane': {'N': 120, 'P': 50, 'K': 60, 'temperature': 28, 'humidity': 75, 'ph': 7.0, 'rainfall': 120, 'season': 'Kharif'},
            'tomato': {'N': 100, 'P': 45, 'K': 50, 'temperature': 24, 'humidity': 70, 'ph': 6.5, 'rainfall': 60, 'season': 'Rabi'},
            'potato': {'N': 90, 'P': 35, 'K': 40, 'temperature': 18, 'humidity': 65, 'ph': 6.0, 'rainfall': 80, 'season': 'Rabi'},
            'onion': {'N': 80, 'P': 30, 'K': 35, 'temperature': 22, 'humidity': 60, 'ph': 6.5, 'rainfall': 70, 'season': 'Rabi'},
            'soybean': {'N': 40, 'P': 20, 'K': 20, 'temperature': 26, 'humidity': 70, 'ph': 6.5, 'rainfall': 100, 'season': 'Kharif'},
            'groundnut': {'N': 30, 'P': 15, 'K': 15, 'temperature': 28, 'humidity': 65, 'ph': 6.8, 'rainfall': 80, 'season': 'Kharif'}
        }
        self.load_or_train_model()

    def load_or_train_model(self):
        try:
            import joblib
            self.model = joblib.load('models/crop_recommendation_model.pkl')
            self.label_encoder = joblib.load('models/crop_label_encoder.pkl')
        except:
            self.train_model()

    def train_model(self):
        try:
            # Generate synthetic data based on ideal conditions
            data = []
            for crop, params in self.crop_data.items():
                for _ in range(100):
                    row = {}
                    for k, v in params.items():
                        if k == 'season':
                            row[k] = v  # Keep season as string
                        else:
                            row[k] = np.random.normal(v, v*0.1)  # Apply noise only to numeric values
                    row['crop'] = crop
                    data.append(row)

            df = pd.DataFrame(data)
            X = df[self.feature_columns]
            y = df['crop']

            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y_encoded)

            # Save the model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/crop_recommendation_model.pkl')
            joblib.dump(self.label_encoder, 'models/crop_label_encoder.pkl')
        except ImportError:
            # sklearn not available, model will remain None and we'll use rule-based
            print("Warning: sklearn not available, using rule-based crop recommendation only")
            self.model = None
            self.label_encoder = None

    def get_weather_data(self, location):
        """Get weather data from NASA POWER API"""
        try:
            # NASA POWER API endpoint for location data
            base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                'parameters': 'T2M,PRECTOT,T2M_MAX,T2M_MIN,RH2M,TS',
                'community': 'RE',
                'longitude': '-0.1257',  # Default to London for demo
                'latitude': '51.5085',
                'start': '2020',
                'end': '2025',
                'format': 'JSON'
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['properties']['parameter']['T2M']['2025'][-1] - 273.15,  # Convert from Kelvin to Celsius
                    'humidity': data['properties']['parameter']['RH2M']['2025'][-1],
                    'rainfall': data['properties']['parameter']['PRECTOT']['2025'][-1] * 30  # Convert to monthly mm
                }
        except:
            pass

        # Return default values if API fails
        return {
            'temperature': 25,
            'humidity': 70,
            'rainfall': 100
        }

    def rule_based_recommendation(self, soil_data, location):
        """Rule-based crop recommendation based on conditions"""
        recommendations = []

        for crop, params in self.crop_data.items():
            score = 0

            # Check temperature suitability
            temp_diff = abs(soil_data.get('temperature', 25) - params['temperature'])
            if temp_diff <= 5:
                score += 3
            elif temp_diff <= 10:
                score += 2
            elif temp_diff <= 15:
                score += 1

            # Check pH suitability
            ph = soil_data.get('ph', 7.0)
            if params['ph'] - 0.5 <= ph <= params['ph'] + 0.5:
                score += 3
            elif params['ph'] - 1 <= ph <= params['ph'] + 1:
                score += 2
            elif params['ph'] - 1.5 <= ph <= params['ph'] + 1.5:
                score += 1

            # Check rainfall suitability
            rainfall = soil_data.get('rainfall', 100)
            rainfall_diff = abs(rainfall - params['rainfall'])
            if rainfall_diff <= 20:
                score += 3
            elif rainfall_diff <= 40:
                score += 2
            elif rainfall_diff <= 60:
                score += 1

            if score >= 6:
                recommendations.append({
                    'crop': crop,
                    'score': score,
                    'season': params['season'],
                    'confidence': min(score * 15, 95)  # Convert score to percentage
                })

        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]

    def recommend_crops(self, soil_data, location):
        """Main method to recommend crops"""
        # Get weather data if location is provided
        if location and location.strip():
            weather_data = self.get_weather_data(location)
            soil_data.update(weather_data)

        # Try ML model first
        if self.model and self.label_encoder:
            try:
                features = np.array([[soil_data.get(col, 0) for col in self.feature_columns]])
                prediction = self.model.predict_proba(features)
                top_indices = prediction[0].argsort()[-3:][::-1]

                ml_recommendations = []
                for idx in top_indices:
                    crop = self.label_encoder.inverse_transform([idx])[0]
                    confidence = prediction[0][idx] * 100
                    if confidence > 10:  # Only include if confidence > 10%
                        ml_recommendations.append({
                            'crop': crop,
                            'confidence': round(confidence, 1),
                            'method': 'ML Model',
                            'season': self.crop_data[crop]['season']
                        })

                if ml_recommendations:
                    return ml_recommendations
            except Exception as e:
                print(f"ML model failed: {e}")

class DiseaseDetector:
    def __init__(self):
        # Rule-based disease database for selected crops
        self.disease_rules = [
            # Rice
            {
                "crop": "Rice",
                "symptoms": ["yellow leaves", "brown spots", "stunted growth", "leaf rot", "whitish powder"],
                "disease": "Brown Spot",
                "pesticide": "Mancozeb",
                "cost_per_ha": 100
            },
            {
                "crop": "Rice",
                "symptoms": ["leaf blight", "yellow stripes", "necrotic lesions", "wilt", "dry tips"],
                "disease": "Bacterial Leaf Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 120
            },
            # Wheat
            {
                "crop": "Wheat",
                "symptoms": ["yellow streaks", "leaf rust", "red pustules", "brown spots", "premature leaf death"],
                "disease": "Leaf Rust",
                "pesticide": "Propiconazole",
                "cost_per_ha": 90
            },
            {
                "crop": "Wheat",
                "symptoms": ["white powdery leaves", "stunted growth", "deformed spikes", "yellow leaves", "reduced yield"],
                "disease": "Powdery Mildew",
                "pesticide": "Sulfur",
                "cost_per_ha": 80
            },
            # Maize
            {
                "crop": "Maize",
                "symptoms": ["yellow leaves", "cob rot", "brown lesions", "brown spots", "stunted growth", "wilting"],
                "disease": "Maize Streak Virus",
                "pesticide": "Imidacloprid",
                "cost_per_ha": 110
            },
            # Cotton
            {
                "crop": "Cotton",
                "symptoms": ["leaf spots", "yellowing", "boll rot", "wilt", "defoliation"],
                "disease": "Bacterial Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 130
            },
            # Sugarcane
            {
                "crop": "Sugarcane",
                "symptoms": ["reddish streaks", "wilting", "brown spots", "stunted growth", "leaf curling"],
                "disease": "Red Rot",
                "pesticide": "Carbendazim",
                "cost_per_ha": 150
            },
            # Tomato
            {
                "crop": "Tomato",
                "symptoms": ["yellow leaves", "spots on leaves", "wilting", "fruit rot", "leaf curl"],
                "disease": "Early Blight",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 120
            },
            # Potato
            {
                "crop": "Potato",
                "symptoms": ["brown spots", "leaf curl", "wilting", "tuber rot", "yellow leaves"],
                "disease": "Late Blight",
                "pesticide": "Metalaxyl",
                "cost_per_ha": 110
            },
            # Onion
            {
                "crop": "Onion",
                "symptoms": ["yellow leaves", "neck rot", "leaf spots", "wilting", "soft rot"],
                "disease": "Purple Blotch",
                "pesticide": "Mancozeb",
                "cost_per_ha": 95
            },
            # Soybean
            {
                "crop": "Soybean",
                "symptoms": ["yellow leaves", "leaf spots", "wilt", "stem lesions", "defoliation"],
                "disease": "Soybean Rust",
                "pesticide": "Trifloxystrobin",
                "cost_per_ha": 100
            },
            # Groundnut
            {
                "crop": "Groundnut",
                "symptoms": ["yellow leaves", "leaf spots", "wilting", "pod rot", "stunted growth"],
                "disease": "Leaf Spot",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 85
            }
        ]

    def detect_disease(self, crop, symptoms):
        """Detect diseases based on symptoms for a given crop"""
        detected = []
        for rule in self.disease_rules:
            if rule["crop"].lower() == crop.lower():
                rule_symptoms = [s.strip().lower() for s in rule["symptoms"]]
                matched = []
                for s in symptoms:
                    for rs in rule_symptoms:
                        if set(s.split()) & set(rs.split()):  # Check for common words
                            matched.append(s)
                            break
                if len(matched) > 0:
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": matched
                    })
        # Fallback: if no matches, return the first disease for the crop
        if not detected:
            for rule in self.disease_rules:
                if rule["crop"].lower() == crop.lower():
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": []  # No matched symptoms
                    })
                    break
        return detected

    def causal_ai_analysis(self, crop, results):
        """Generate detailed causal analysis for detected diseases using disease_data"""
        analysis = []
        for r in results:
            pesticide = r.get('pesticide', 'Consult expert for appropriate pesticide')
            cost = r.get('cost_per_ha', 0)
            disease = r['disease']
            
            # Get detailed disease info if available
            disease_info = self.disease_data.get(disease.lower().replace(' ', '_'), {})
            
            para = f"""
Causal Analysis for {crop}:

The disease '{disease}' was detected primarily due to the presence of symptoms {r['matched_symptoms']}. 
These symptoms indicate stress or infection in the crop that, if left unchecked, could lead to significant yield loss, stunted growth, or complete crop failure. 

Immediate application of the recommended pesticide '{pesticide}' at a cost of ${cost} per hectare is advised to mitigate the spread. 
Monitoring and preventive measures should follow to avoid recurrence, ensuring crop health and productivity.

"""
            if disease_info:
                para += f"""
Additional Insights:
- **Type**: {disease_info.get('type', 'Unknown')}
- **Causes**: {', '.join(disease_info.get('causes', []))}
- **Effects**: {disease_info.get('effects', 'Reduced crop health and productivity')}
- **Prevention**: {', '.join(disease_info.get('prevention', []))}
- **Spread Rate**: {disease_info.get('spread_rate', 'Unknown')}
- **Economic Impact**: {disease_info.get('economic_impact', 'Variable')}
- **Treatment Cost**: {disease_info.get('treatment_cost', 'Variable')}
"""
            analysis.append(para.strip())
        return analysis
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import requests
from datetime import datetime
import os

class CropRecommender:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_data = {
            'rice': {'N': 80, 'P': 40, 'K': 40, 'temperature': 25, 'humidity': 80, 'ph': 6.5, 'rainfall': 150, 'season': 'Kharif'},
            'wheat': {'N': 70, 'P': 30, 'K': 30, 'temperature': 20, 'humidity': 60, 'ph': 7.0, 'rainfall': 100, 'season': 'Rabi'},
            'maize': {'N': 90, 'P': 40, 'K': 35, 'temperature': 27, 'humidity': 70, 'ph': 6.0, 'rainfall': 90, 'season': 'Kharif'},
            'cotton': {'N': 60, 'P': 25, 'K': 25, 'temperature': 30, 'humidity': 65, 'ph': 7.5, 'rainfall': 80, 'season': 'Kharif'},
            'sugarcane': {'N': 120, 'P': 50, 'K': 60, 'temperature': 28, 'humidity': 75, 'ph': 7.0, 'rainfall': 120, 'season': 'Kharif'},
            'tomato': {'N': 100, 'P': 45, 'K': 50, 'temperature': 24, 'humidity': 70, 'ph': 6.5, 'rainfall': 60, 'season': 'Rabi'},
            'potato': {'N': 90, 'P': 35, 'K': 40, 'temperature': 18, 'humidity': 65, 'ph': 6.0, 'rainfall': 80, 'season': 'Rabi'},
            'onion': {'N': 80, 'P': 30, 'K': 35, 'temperature': 22, 'humidity': 60, 'ph': 6.5, 'rainfall': 70, 'season': 'Rabi'},
            'soybean': {'N': 40, 'P': 20, 'K': 20, 'temperature': 26, 'humidity': 70, 'ph': 6.5, 'rainfall': 100, 'season': 'Kharif'},
            'groundnut': {'N': 30, 'P': 15, 'K': 15, 'temperature': 28, 'humidity': 65, 'ph': 6.8, 'rainfall': 80, 'season': 'Kharif'}
        }
        self.load_or_train_model()

    def load_or_train_model(self):
        try:
            import joblib
            self.model = joblib.load('models/crop_recommendation_model.pkl')
            self.label_encoder = joblib.load('models/crop_label_encoder.pkl')
        except:
            self.train_model()

    def train_model(self):
        try:
            # Generate synthetic data based on ideal conditions
            data = []
            for crop, params in self.crop_data.items():
                for _ in range(100):
                    row = {}
                    for k, v in params.items():
                        if k == 'season':
                            row[k] = v  # Keep season as string
                        else:
                            row[k] = np.random.normal(v, v*0.1)  # Apply noise only to numeric values
                    row['crop'] = crop
                    data.append(row)

            df = pd.DataFrame(data)
            X = df[self.feature_columns]
            y = df['crop']

            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y_encoded)

            # Save the model
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/crop_recommendation_model.pkl')
            joblib.dump(self.label_encoder, 'models/crop_label_encoder.pkl')
        except ImportError:
            # sklearn not available, model will remain None and we'll use rule-based
            print("Warning: sklearn not available, using rule-based crop recommendation only")
            self.model = None
            self.label_encoder = None

    def get_weather_data(self, location):
        """Get weather data from NASA POWER API"""
        try:
            # NASA POWER API endpoint for location data
            base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                'parameters': 'T2M,PRECTOT,T2M_MAX,T2M_MIN,RH2M,TS',
                'community': 'RE',
                'longitude': '-0.1257',  # Default to London for demo
                'latitude': '51.5085',
                'start': '2020',
                'end': '2025',
                'format': 'JSON'
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['properties']['parameter']['T2M']['2025'][-1] - 273.15,  # Convert from Kelvin to Celsius
                    'humidity': data['properties']['parameter']['RH2M']['2025'][-1],
                    'rainfall': data['properties']['parameter']['PRECTOT']['2025'][-1] * 30  # Convert to monthly mm
                }
        except:
            pass

        # Return default values if API fails
        return {
            'temperature': 25,
            'humidity': 70,
            'rainfall': 100
        }

    def rule_based_recommendation(self, soil_data, location):
        """Rule-based crop recommendation based on conditions"""
        recommendations = []

        for crop, params in self.crop_data.items():
            score = 0

            # Check temperature suitability
            temp_diff = abs(soil_data.get('temperature', 25) - params['temperature'])
            if temp_diff <= 5:
                score += 3
            elif temp_diff <= 10:
                score += 2
            elif temp_diff <= 15:
                score += 1

            # Check pH suitability
            ph = soil_data.get('ph', 7.0)
            if params['ph'] - 0.5 <= ph <= params['ph'] + 0.5:
                score += 3
            elif params['ph'] - 1 <= ph <= params['ph'] + 1:
                score += 2
            elif params['ph'] - 1.5 <= ph <= params['ph'] + 1.5:
                score += 1

            # Check rainfall suitability
            rainfall = soil_data.get('rainfall', 100)
            rainfall_diff = abs(rainfall - params['rainfall'])
            if rainfall_diff <= 20:
                score += 3
            elif rainfall_diff <= 40:
                score += 2
            elif rainfall_diff <= 60:
                score += 1

            if score >= 6:
                recommendations.append({
                    'crop': crop,
                    'score': score,
                    'season': params['season'],
                    'confidence': min(score * 15, 95)  # Convert score to percentage
                })

        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]

    def recommend_crops(self, soil_data, location):
        """Main method to recommend crops"""
        # Get weather data if location is provided
        if location and location.strip():
            weather_data = self.get_weather_data(location)
            soil_data.update(weather_data)

        # Try ML model first
        if self.model and self.label_encoder:
            try:
                features = np.array([[soil_data.get(col, 0) for col in self.feature_columns]])
                prediction = self.model.predict_proba(features)
                top_indices = prediction[0].argsort()[-3:][::-1]

                ml_recommendations = []
                for idx in top_indices:
                    crop = self.label_encoder.inverse_transform([idx])[0]
                    confidence = prediction[0][idx] * 100
                    if confidence > 10:  # Only include if confidence > 10%
                        ml_recommendations.append({
                            'crop': crop,
                            'confidence': round(confidence, 1),
                            'method': 'ML Model',
                            'season': self.crop_data[crop]['season']
                        })

                if ml_recommendations:
                    return ml_recommendations
            except Exception as e:
                print(f"ML model failed: {e}")

class DiseaseDetector:
    def __init__(self):
        # Rule-based disease database for selected crops
        self.disease_rules = [
            # Rice
            {
                "crop": "Rice",
                "symptoms": ["yellow leaves", "brown spots", "stunted growth", "leaf rot", "whitish powder"],
                "disease": "Brown Spot",
                "pesticide": "Mancozeb",
                "cost_per_ha": 100
            },
            {
                "crop": "Rice",
                "symptoms": ["leaf blight", "yellow stripes", "necrotic lesions", "wilt", "dry tips"],
                "disease": "Bacterial Leaf Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 120
            },
            # Wheat
            {
                "crop": "Wheat",
                "symptoms": ["yellow streaks", "leaf rust", "red pustules", "brown spots", "premature leaf death"],
                "disease": "Leaf Rust",
                "pesticide": "Propiconazole",
                "cost_per_ha": 90
            },
            {
                "crop": "Wheat",
                "symptoms": ["white powdery leaves", "stunted growth", "deformed spikes", "yellow leaves", "reduced yield"],
                "disease": "Powdery Mildew",
                "pesticide": "Sulfur",
                "cost_per_ha": 80
            },
            # Maize
            {
                "crop": "Maize",
                "symptoms": ["yellow leaves", "cob rot", "brown lesions", "brown spots", "stunted growth", "wilting"],
                "disease": "Maize Streak Virus",
                "pesticide": "Imidacloprid",
                "cost_per_ha": 110
            },
            # Cotton
            {
                "crop": "Cotton",
                "symptoms": ["leaf spots", "yellowing", "boll rot", "wilt", "defoliation"],
                "disease": "Bacterial Blight",
                "pesticide": "Copper Oxychloride",
                "cost_per_ha": 130
            },
            # Sugarcane
            {
                "crop": "Sugarcane",
                "symptoms": ["reddish streaks", "wilting", "brown spots", "stunted growth", "leaf curling"],
                "disease": "Red Rot",
                "pesticide": "Carbendazim",
                "cost_per_ha": 150
            },
            # Tomato
            {
                "crop": "Tomato",
                "symptoms": ["yellow leaves", "spots on leaves", "wilting", "fruit rot", "leaf curl"],
                "disease": "Early Blight",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 120
            },
            # Potato
            {
                "crop": "Potato",
                "symptoms": ["brown spots", "leaf curl", "wilting", "tuber rot", "yellow leaves"],
                "disease": "Late Blight",
                "pesticide": "Metalaxyl",
                "cost_per_ha": 110
            },
            # Onion
            {
                "crop": "Onion",
                "symptoms": ["yellow leaves", "neck rot", "leaf spots", "wilting", "soft rot"],
                "disease": "Purple Blotch",
                "pesticide": "Mancozeb",
                "cost_per_ha": 95
            },
            # Soybean
            {
                "crop": "Soybean",
                "symptoms": ["yellow leaves", "leaf spots", "wilt", "stem lesions", "defoliation"],
                "disease": "Soybean Rust",
                "pesticide": "Trifloxystrobin",
                "cost_per_ha": 100
            },
            # Groundnut
            {
                "crop": "Groundnut",
                "symptoms": ["yellow leaves", "leaf spots", "wilting", "pod rot", "stunted growth"],
                "disease": "Leaf Spot",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 85
            }
        ]

    def detect_disease(self, crop, symptoms):
        """Detect diseases based on symptoms for a given crop"""
        detected = []
        for rule in self.disease_rules:
            if rule["crop"].lower() == crop.lower():
                rule_symptoms = [s.strip().lower() for s in rule["symptoms"]]
                matched = []
                for s in symptoms:
                    for rs in rule_symptoms:
                        if set(s.split()) & set(rs.split()):  # Check for common words
                            matched.append(s)
                            break
                if len(matched) > 0:
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": matched
                    })
        # Fallback: if no matches, return the first disease for the crop
        if not detected:
            for rule in self.disease_rules:
                if rule["crop"].lower() == crop.lower():
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": []  # No matched symptoms
                    })
                    break
        return detected

    def causal_ai_analysis(self, crop, results):
        """Generate detailed causal analysis for detected diseases using disease_data"""
        analysis = []
        for r in results:
            pesticide = r.get('pesticide', 'Consult expert for appropriate pesticide')
            cost = r.get('cost_per_ha', 0)
            disease = r['disease']
            
            # Get detailed disease info if available
            disease_info = self.disease_data.get(disease.lower().replace(' ', '_'), {})
            
            para = f"""
Causal Analysis for {crop}:

The disease '{disease}' was detected primarily due to the presence of symptoms {r['matched_symptoms']}. 
These symptoms indicate stress or infection in the crop that, if left unchecked, could lead to significant yield loss, stunted growth, or complete crop failure. 

Immediate application of the recommended pesticide '{pesticide}' at a cost of ${cost} per hectare is advised to mitigate the spread. 
Monitoring and preventive measures should follow to avoid recurrence, ensuring crop health and productivity.

"""
            if disease_info:
                para += f"""
Additional Insights:
- **Type**: {disease_info.get('type', 'Unknown')}
- **Causes**: {', '.join(disease_info.get('causes', []))}
- **Effects**: {disease_info.get('effects', 'Reduced crop health and productivity')}
- **Prevention**: {', '.join(disease_info.get('prevention', []))}
- **Spread Rate**: {disease_info.get('spread_rate', 'Unknown')}
- **Economic Impact**: {disease_info.get('economic_impact', 'Variable')}
- **Treatment Cost**: {disease_info.get('treatment_cost', 'Variable')}
"""
            analysis.append(para.strip())
        return analysis
    def __init__(self):
        # Sample rule-based crop disease database
        self.disease_rules = [
            {
                "crop": "Tomato",
                "symptoms": ["yellow leaves", "spots on leaves", "wilting"],
                "disease": "Early Blight",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 120
            },
            {
                "crop": "Wheat",
                "symptoms": ["yellow streaks", "leaf rust"],
                "disease": "Leaf Rust",
                "pesticide": "Propiconazole",
                "cost_per_ha": 90
            },
            {
                "crop": "Rice",
                "symptoms": ["brown spots", "stunted growth"],
                "disease": "Brown Spot",
                "pesticide": "Mancozeb",
                "cost_per_ha": 100
            }
        ]

        self.disease_data = {
            'early_blight': {
                'type': 'Fungal',
                'crop_affected': ['tomato', 'potato', 'pepper', 'eggplant'],
                'symptoms': ['dark spots with concentric rings', 'yellow halos around spots', 'leaf curling', 'brown lesions', 'target-like spots on leaves', 'spots starting from lower leaves', 'irregular shaped lesions'],
                'causes': ['Alternaria solani fungus', 'warm humid conditions (20-30°C)', 'poor air circulation', 'overcrowded plants', 'prolonged leaf wetness'],
                'effects': ['reduced photosynthesis by 50-70%', 'premature leaf drop', 'reduced yield by 20-50%', 'plant death in severe cases', 'fruit quality deterioration'],
                'prevention': ['crop rotation every 2-3 years', 'proper plant spacing (45-60cm)', 'fungicide application before symptoms', 'remove and destroy infected leaves immediately', 'avoid overhead watering', 'use drip irrigation'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ],
                            'potato': [
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'},
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'}
                            ],
                            'pepper': [
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ],
                            'eggplant': [
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'},
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'potato': [
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'},
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'}
                            ],
                            'pepper': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'eggplant': [
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'},
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'potato': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ],
                            'pepper': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'eggplant': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['spot size > 1cm diameter', 'multiple spots per leaf (>5)', 'spots on upper leaves', 'rapid spot expansion', 'defoliation > 25%'],
                'visual_cues': ['concentric ring pattern', 'yellow halo around dark center', 'irregular lesion shape'],
                'spread_rate': 'Moderate to High',
                'economic_impact': 'High (20-50% yield loss)',
                'treatment_cost': 'Medium ($50-150/ha)'
            },
            'late_blight': {
                'type': 'Fungal',
                'crop_affected': ['tomato', 'potato'],
                'symptoms': ['water-soaked lesions on leaves', 'white fungal growth on leaf undersides', 'rapid plant death within days', 'dark brown spots with greasy appearance', 'rapid defoliation', 'stem lesions', 'fruit rot'],
                'causes': ['Phytophthora infestans fungus', 'cool moist conditions (15-25°C)', 'high humidity (>80%)', 'poor drainage', 'prolonged leaf wetness (>6 hours)', 'infected seed tubers'],
                'effects': ['complete crop loss within 2-3 days under favorable conditions', 'rapid spore spread to nearby plants', 'contamination of tubers and soil', 'economic loss up to 100%', 'secondary bacterial infections'],
                'prevention': ['use resistant varieties (ex: Kufri Jyoti, Kufri Himalini)', 'proper irrigation management', 'good field sanitation', 'avoid overhead watering', 'maintain proper plant spacing', 'apply preventive fungicides', 'early planting to avoid peak humidity'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 7-10 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'},
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'}
                            ],
                            'potato': [
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 5-7 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'potato': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Dimethomorph 50% WP', 'dosage': '1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹1200/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Emergency treatment every 3-5 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Fluopicolide 4.44% + Fosetyl-Al 66.7% WG', 'dosage': '2.5g/L', 'price_per_kg': 1500, 'estimated_cost': '₹3750/ha'}
                            ],
                            'potato': [
                                {'name': 'Fluazinam 50% WP', 'dosage': '1g/L', 'price_per_kg': 1800, 'estimated_cost': '₹1800/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 850, 'estimated_cost': '₹2125/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['rapid spread rate', 'white mold on leaf undersides', 'plant collapse within 48 hours', 'stem infection', 'multiple plants affected'],
                'visual_cues': ['white cottony growth on leaf undersides', 'water-soaked appearance', 'rapid lesion expansion'],
                'spread_rate': 'Very High (explosive outbreaks)',
                'economic_impact': 'Severe (50-100% crop loss)',
                'treatment_cost': 'High ($100-300/ha)'
            },
            'powdery_mildew': {
                'type': 'Fungal',
                'crop_affected': ['cucumber', 'melon', 'pumpkin', 'grape', 'rose', 'pea'],
                'symptoms': ['white powdery coating on leaf surfaces', 'yellowing of infected leaves', 'stunted growth', 'distorted leaves and shoots', 'premature leaf drop', 'reduced fruit quality', 'white patches that can be rubbed off'],
                'causes': ['Erysiphe cichoracearum and other Erysiphe species', 'high humidity (60-80%) with moderate temperatures (20-25°C)', 'poor air circulation', 'nitrogen deficiency', 'dense plant canopy', 'overhead irrigation'],
                'effects': ['reduced photosynthesis by 30-50%', 'lower quality produce', 'yield loss up to 30%', 'reduced plant vigor', 'increased susceptibility to other diseases', 'premature plant death'],
                'prevention': ['proper plant spacing for air circulation', 'avoid overhead irrigation', 'resistant varieties selection', 'balanced fertilization (avoid excess nitrogen)', 'good sanitation practices', 'remove infected plant debris', 'improve ventilation'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'cucumber': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Potassium bicarbonate', 'dosage': '5g/L', 'price_per_kg': 200, 'estimated_cost': '₹1000/ha'}
                            ],
                            'melon': [
                                {'name': 'Wettable sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Neem oil 1%', 'dosage': '5ml/L', 'price_per_liter': 350, 'estimated_cost': '₹1750/ha'}
                            ],
                            'pumpkin': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Potassium bicarbonate', 'dosage': '5g/L', 'price_per_kg': 200, 'estimated_cost': '₹1000/ha'}
                            ],
                            'grape': [
                                {'name': 'Sulfur 80% WP', 'dosage': '2.5g/L', 'price_per_kg': 120, 'estimated_cost': '₹300/ha'},
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'}
                            ],
                            'rose': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ],
                            'pea': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Potassium bicarbonate', 'dosage': '5g/L', 'price_per_kg': 200, 'estimated_cost': '₹1000/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'cucumber': [
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ],
                            'melon': [
                                {'name': 'Hexaconazole 5% EC', 'dosage': '1ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹2200/ha'},
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'}
                            ],
                            'pumpkin': [
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ],
                            'grape': [
                                {'name': 'Flusilazole 40% EC', 'dosage': '0.25ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹800/ha'},
                                {'name': 'Penconazole 10% EC', 'dosage': '0.5ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹1400/ha'}
                            ],
                            'rose': [
                                {'name': 'Hexaconazole 5% EC', 'dosage': '1ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹2200/ha'},
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'}
                            ],
                            'pea': [
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'cucumber': [
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'melon': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Kresoxim-methyl 44.3% SC', 'dosage': '0.5ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹1750/ha'}
                            ],
                            'pumpkin': [
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'grape': [
                                {'name': 'Trifloxystrobin 25% + Tebuconazole 50% WG', 'dosage': '0.5g/L', 'price_per_kg': 2200, 'estimated_cost': '₹1100/ha'},
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'}
                            ],
                            'rose': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Kresoxim-methyl 44.3% SC', 'dosage': '0.5ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹1750/ha'}
                            ],
                            'pea': [
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['powdery coating density (>50% leaf coverage)', 'leaf yellowing extent', 'growth stunting (>20%)', 'multiple plant infection', 'early season occurrence'],
                'visual_cues': ['white powdery coating', 'can be easily rubbed off', 'starts as small white spots'],
                'spread_rate': 'High (wind and water dispersed)',
                'economic_impact': 'Medium (15-30% yield loss)',
                'treatment_cost': 'Low to Medium ($30-80/ha)'
            },
            'bacterial_spot': {
                'type': 'Bacterial',
                'crop_affected': ['tomato', 'pepper'],
                'symptoms': ['small water-soaked spots on leaves', 'yellow halos around spots', 'leaf perforation as spots age', 'dark raised lesions on fruit', 'severe defoliation in humid conditions', 'angular leaf spots', 'spots turn brown and necrotic'],
                'causes': ['Xanthomonas vesicatoria and related species', 'warm wet conditions (25-30°C)', 'infected seeds or transplants', 'splashing water from rain or irrigation', 'high humidity (>85%)', 'wounds from insects or hail'],
                'effects': ['severe defoliation in humid conditions', 'reduced fruit quality and market value', 'yield loss up to 40%', 'secondary infections by other pathogens', 'unmarketable fruit due to spotting'],
                'prevention': ['use disease-free seeds and transplants', 'crop rotation with non-host plants (3 years)', 'avoid overhead watering', 'copper-based preventive sprays', 'good field sanitation', 'remove volunteer plants', 'use resistant varieties'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ],
                            'pepper': [
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Streptomycin sulfate 90% SP', 'dosage': '0.1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹120/ha'},
                                {'name': 'Kasugamycin 3% SL', 'dosage': '2ml/L', 'price_per_liter': 1800, 'estimated_cost': '₹3600/ha'}
                            ],
                            'pepper': [
                                {'name': 'Streptomycin sulfate 90% SP', 'dosage': '0.1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹120/ha'},
                                {'name': 'Kasugamycin 3% SL', 'dosage': '2ml/L', 'price_per_liter': 1800, 'estimated_cost': '₹3600/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Oxytetracycline hydrochloride 50% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'},
                                {'name': 'Validamycin 3% L', 'dosage': '2ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹4400/ha'}
                            ],
                            'pepper': [
                                {'name': 'Oxytetracycline hydrochloride 50% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'},
                                {'name': 'Validamycin 3% L', 'dosage': '2ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹4400/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['halo formation around spots', 'rapid spot expansion (>1mm/day)', 'leaf perforation', 'fruit infection', 'multiple plants affected'],
                'visual_cues': ['angular spots (limited by leaf veins)', 'water-soaked appearance initially', 'yellow halos'],
                'spread_rate': 'High (rain splash dispersed)',
                'economic_impact': 'High (30-50% yield loss)',
                'treatment_cost': 'Medium ($60-120/ha)'
            },
            'bacterial_wilt': {
                'type': 'Bacterial',
                'crop_affected': ['tomato', 'pepper', 'potato', 'eggplant'],
                'symptoms': ['sudden wilting of entire plant', 'yellowing and browning of leaves', 'vascular discoloration (brown streaks)', 'plant death within days', 'no response to watering', 'stunted growth', 'adventitious roots on stems'],
                'causes': ['Ralstonia solanacearum bacteria', 'warm soil temperatures (25-35°C)', 'high soil moisture', 'poor soil drainage', 'infected soil or water', 'mechanical damage to roots', 'monoculture practices'],
                'effects': ['complete plant death', 'soil contamination for multiple seasons', 'yield loss up to 80%', 'reduced crop quality', 'economic losses from replanting', 'spread to adjacent fields'],
                'prevention': ['use resistant varieties', 'crop rotation with non-solanaceous crops', 'soil solarization', 'improve soil drainage', 'use clean irrigation water', 'avoid root damage during cultivation', 'soil testing for pathogen presence'],
                'treatment': ['no effective chemical control available', 'remove infected plants immediately with soil', 'soil fumigation for future crops', 'improve soil drainage', 'use resistant varieties', 'avoid susceptible crops for 2-3 years'],
                'severity_indicators': ['vascular browning in stems', 'rapid wilting despite adequate moisture', 'plant collapse within 3-5 days', 'multiple plants affected suddenly'],
                'visual_cues': ['brown vascular discoloration', 'hollow stem appearance', 'bacterial ooze from cut stems'],
                'spread_rate': 'Very High (soil and water borne)',
                'economic_impact': 'Severe (50-90% crop loss)',
                'treatment_cost': 'High (soil remediation costs)'
            },

            # Viral Diseases
            'tomato_yellow_leaf_curl': {
                'type': 'Viral',
                'crop_affected': ['tomato'],
                'symptoms': ['upward curling of leaves', 'yellowing between veins', 'stunted plant growth', 'reduced leaf size', 'flower drop', 'deformed fruit', 'purple tinged leaves', 'shortened internodes'],
                'causes': ['Tomato yellow leaf curl virus (TYLCV)', 'whitefly vector (Bemisia tabaci)', 'infected transplants or seeds', 'warm temperatures favoring whitefly reproduction', 'monoculture tomato production', 'poor vector control'],
                'effects': ['stunted plant growth and reduced vigor', 'significant yield reduction (30-70%)', 'poor fruit quality and size', 'delayed maturity', 'increased susceptibility to other stresses', 'economic losses from reduced marketable yield'],
                'prevention': ['use virus-free transplants and seeds', 'whitefly vector control with insecticides', 'use reflective mulches to repel whiteflies', 'remove infected plants immediately', 'crop rotation with non-host crops', 'use resistant varieties', 'early planting to avoid peak whitefly season'],
                'treatment': ['no direct chemical treatment for virus', 'control whitefly vectors with systemic insecticides', 'remove infected plants immediately', 'use insect-proof netting', 'apply mineral oils to disrupt virus transmission', 'maintain field sanitation'],
                'severity_indicators': ['leaf curling >50% of plant', 'stunted growth (<50% normal height)', 'yellow vein clearing', 'multiple plants infected', 'early season infection'],
                'visual_cues': ['severe upward leaf curl', 'yellowing between leaf veins', 'stunted bushy appearance'],
                'spread_rate': 'High (vector transmitted)',
                'economic_impact': 'High (40-70% yield loss)',
                'treatment_cost': 'Medium ($80-150/ha)'
            },

            # Nutrient Deficiencies (often misdiagnosed as diseases)
            'nitrogen_deficiency': {
                'type': 'Nutritional',
                'crop_affected': ['all crops'],
                'symptoms': ['older leaves turn yellow first', 'general yellowing of foliage', 'stunted growth', 'thin stems', 'reduced tillering', 'small leaves', 'premature flowering', 'pale green to yellow coloration'],
                'causes': ['insufficient nitrogen in soil', 'poor soil fertility', 'excessive leaching due to heavy rainfall', 'high soil pH reducing availability', 'crop rotation without nitrogen fixation', 'previous crop nitrogen depletion'],
                'effects': ['reduced photosynthesis and plant vigor', 'lower protein content in grains', 'yield reduction up to 30%', 'increased susceptibility to diseases', 'poor root development', 'delayed maturity'],
                'prevention': ['soil testing before planting', 'proper fertilization with nitrogen sources', 'use of organic matter and compost', 'crop rotation with legumes', 'split application of nitrogen fertilizer', 'avoid over-irrigation that causes leaching'],
                'treatment': ['apply nitrogen fertilizers (urea, ammonium nitrate)', 'foliar application of urea (1-2%)', 'incorporate organic matter', 'use slow-release nitrogen sources', 'ensure proper irrigation', 'adjust soil pH if necessary'],
                'severity_indicators': ['yellowing starting from lower leaves', 'general plant weakness', 'small leaf size', 'reduced growth rate', 'multiple plants affected uniformly'],
                'visual_cues': ['uniform yellowing pattern', 'starts from older leaves', 'no spots or lesions'],
                'spread_rate': 'Uniform across field',
                'economic_impact': 'Medium (15-30% yield loss)',
                'treatment_cost': 'Low ($20-50/ha)'
            },
            'potassium_deficiency': {
                'type': 'Nutritional',
                'crop_affected': ['all crops'],
                'symptoms': ['yellowing and browning of leaf margins', 'scorching of older leaves', 'weak stems', 'small fruit size', 'poor fruit quality', 'leaf curling', 'reduced disease resistance', 'brown spots on leaves'],
                'causes': ['low potassium levels in soil', 'sandy soils with poor cation exchange', 'excessive calcium or magnesium', 'high rainfall causing leaching', 'continuous cropping without replacement', 'acidic soil conditions'],
                'effects': ['reduced plant vigor and stress tolerance', 'poor fruit development and quality', 'yield reduction up to 25%', 'increased disease susceptibility', 'reduced winter hardiness', 'poor root development'],
                'prevention': ['regular soil testing for potassium levels', 'use of potassium-rich fertilizers', 'maintain proper soil pH (6.0-7.0)', 'avoid excessive nitrogen application', 'use organic matter to improve CEC', 'balanced fertilization program'],
                'treatment': ['apply potassium fertilizers (potassium chloride, sulfate)', 'foliar application of potassium nitrate', 'use potassium-rich organic sources', 'improve soil structure with organic matter', 'ensure proper irrigation practices', 'avoid soil compaction'],
                'severity_indicators': ['marginal leaf burn', 'scorching of older leaves', 'weak stem strength', 'poor fruit development', 'uniform field pattern'],
                'visual_cues': ['leaf margin necrosis', 'brown scorching starting from edges', 'interveinal chlorosis'],
                'spread_rate': 'Gradual across field',
                'economic_impact': 'Medium (15-25% yield loss)',
                'treatment_cost': 'Low to Medium ($30-70/ha)'
            },

            # Insect-Related Issues
            'onion_downy_mildew': {
                'type': 'Fungal',
                'crop_affected': ['onion', 'garlic', 'shallot'],
                'symptoms': ['white powdery coating on leaves', 'reduced yield', 'symptoms on lower leaves first', 'yellowing of leaves', 'stunted growth', 'premature leaf dieback'],
                'causes': ['Peronospora destructor fungus', 'cool moist conditions (15-20°C)', 'high humidity (>80%)', 'poor air circulation', 'dense plantings', 'infected seeds or soil'],
                'effects': ['reduced photosynthesis leading to lower bulb size', 'yield loss up to 50%', 'poor bulb quality', 'increased susceptibility to secondary infections', 'complete crop failure in severe cases'],
                'prevention': ['use disease-free seeds', 'crop rotation every 3-4 years', 'proper plant spacing (10-15cm)', 'improve air circulation', 'avoid overhead irrigation', 'remove crop debris after harvest'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'onion': [
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ],
                            'garlic': [
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ],
                            'shallot': [
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'onion': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'garlic': [
                                {'name': 'Dimethomorph 50% WP', 'dosage': '1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹1200/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ],
                            'shallot': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'onion': [
                                {'name': 'Fluopicolide 4.44% + Fosetyl-Al 66.7% WG', 'dosage': '2.5g/L', 'price_per_kg': 1500, 'estimated_cost': '₹3750/ha'},
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'}
                            ],
                            'garlic': [
                                {'name': 'Fluazinam 50% WP', 'dosage': '1g/L', 'price_per_kg': 1800, 'estimated_cost': '₹1800/ha'},
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'}
                            ],
                            'shallot': [
                                {'name': 'Fluopicolide 4.44% + Fosetyl-Al 66.7% WG', 'dosage': '2.5g/L', 'price_per_kg': 1500, 'estimated_cost': '₹3750/ha'},
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['white coating density (>50% leaf coverage)', 'yellowing extent on lower leaves', 'stunted growth (>20%)', 'rapid symptom progression', 'multiple plants affected'],
                'visual_cues': ['white powdery coating on leaf surfaces', 'starts on lower leaves', 'yellowing between veins'],
                'spread_rate': 'High (wind and water dispersed)',
                'economic_impact': 'High (30-50% yield loss)',
                'treatment_cost': 'Medium ($100-200/ha)'
            },
        }

        # Symptom categories for dropdown selection - aligned with disease database
        self.symptom_categories = {
            'leaf_symptoms': [
                'Dark spots with concentric rings',  # Early Blight
                'Yellow halos around spots',        # Early Blight
                'Leaf curling',                     # Early Blight, Viral
                'Brown lesions',                    # Early Blight
                'Target-like spots on leaves',      # Early Blight
                'White powdery coating on leaves',  # Powdery Mildew, Onion Downy Mildew
                'Water-soaked lesions',             # Late Blight, Bacterial Spot
                'Angular leaf spots',               # Bacterial Spot
                'Leaf curling upward',              # Viral diseases
                'Yellowing between leaf veins',     # Viral diseases, Nutrient deficiency
                'Yellowing starting from leaf margins', # Potassium deficiency
                'Premature leaf drop',              # Multiple diseases
                'Stunted leaf growth',              # Multiple diseases
                'White mold on leaf undersides',    # Late Blight
                'Greasy appearance on leaves',      # Late Blight
                'Symptoms on lower leaves first'    # Onion Downy Mildew
            ],
            'plant_symptoms': [
                'Sudden wilting of entire plant',   # Bacterial Wilt
                'Stunted plant growth',             # Viral, Nutrient deficiencies
                'Plant death within days',          # Late Blight, Bacterial Wilt
                'Flower drop',                      # Viral diseases
                'Deformed fruit',                   # Viral diseases, Bacterial Spot
                'Small fruit size',                 # Nutrient deficiencies
                'Purple tinged leaves',             # Viral diseases
                'No response to watering',          # Bacterial Wilt
                'Reduced yield'                     # Onion Downy Mildew
            ],
            'environmental_indicators': [
                'Symptoms worse during humid periods',
                'Symptoms appear after rainfall',
                'Symptoms worse in warm conditions',
                'Symptoms worse in cool conditions',
                'Rapid symptom progression',
                'Symptoms spreading to nearby plants',
                'Symptoms on lower leaves first',
                'Symptoms on upper leaves first'
            ]
        }

        self.load_or_train_model()

    def load_or_train_model(self):
        """Load or train the disease detection model"""
        try:
            import joblib
            self.model = joblib.load('models/disease_detection_model.pkl')
            self.label_encoder = joblib.load('models/disease_label_encoder.pkl')
        except (ImportError, FileNotFoundError, IOError):
            # joblib not available or model files don't exist, use rule-based only
            # In a real implementation, you would train a CNN model here
            self.model = None
            self.label_encoder = None

    def extract_image_features(self, image_path):
        """Extract features from crop leaf image"""
        try:
            from PIL import Image
            import cv2
            import numpy as np

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize for consistent processing
            img_resized = cv2.resize(img_rgb, (224, 224))

            # Extract color features (mean and std of each channel)
            features = []
            for channel in range(3):  # R, G, B
                channel_data = img_resized[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data)
                ])

            # Extract texture features using gray scale
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

            # Calculate texture features
            features.extend([
                np.mean(gray),  # Mean intensity
                np.std(gray),   # Standard deviation
                np.var(gray)    # Variance
            ])

            # Calculate edges using Sobel
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Edge magnitude
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features.extend([
                np.mean(edge_magnitude),
                np.std(edge_magnitude)
            ])

            # HSV color space features
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
            for channel in range(3):  # H, S, V
                channel_data = hsv[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data)
                ])

            return np.array(features).reshape(1, -1)

        except ImportError:
            print("Warning: Image processing libraries (PIL, cv2) not available")
            return None
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def rule_based_detection(self, crop_type, selected_symptoms, severity, image_features=None):
        """Enhanced rule-based disease detection with comprehensive symptom matching"""
        matches = []

        # Debug: Print what symptoms we're receiving
        print(f"DEBUG: Crop type: {crop_type}")
        print(f"DEBUG: Selected symptoms: {selected_symptoms}")
        print(f"DEBUG: Severity: {severity}")

        # Normalize crop type
        crop_type_lower = crop_type.lower().strip() if crop_type else ""

        for disease, data in self.disease_data.items():
            score = 0
            reasoning = []
            matched_symptoms = []

            # Check if crop is affected by this disease (case-insensitive and flexible matching)
            affected_crops_lower = [c.lower().strip() for c in data['crop_affected']]

            # More flexible crop matching
            crop_matched = False
            for affected_crop in affected_crops_lower:
                if (affected_crop in crop_type_lower or
                    crop_type_lower in affected_crop or
                    affected_crop.replace(' ', '') in crop_type_lower.replace(' ', '') or
                    crop_type_lower.replace(' ', '') in affected_crop.replace(' ', '')):
                    crop_matched = True
                    break

            if not crop_matched:
                print(f"DEBUG: Skipping {disease} - not for crop {crop_type}")
                continue

            print(f"DEBUG: Checking disease: {disease}")

            # If no symptoms provided, give basic matching score
            if not selected_symptoms or len(selected_symptoms) == 0:
                score += 15  # Base score for crop match
                reasoning.append(f"✅ Disease commonly affects {crop_type.title()}")
                matches.append({
                    'disease': disease,
                    'confidence': min(score, 95),
                    'details': data,
                    'method': 'Rule-based Analysis',
                    'reasoning': reasoning[:2],
                    'matched_symptoms': [],
                    'type': data['type']
                })
                continue

            # Symptom matching with categories - more flexible matching
            for selected_symptom in selected_symptoms:
                symptom_lower = selected_symptom.lower().strip()
                print(f"DEBUG: Checking symptom: '{selected_symptom}'")

                # Check against disease symptoms with flexible matching
                for disease_symptom in data['symptoms']:
                    disease_symptom_lower = disease_symptom.lower()
                    print(f"DEBUG: Comparing with disease symptom: '{disease_symptom}'")

                    # More flexible matching - check for key terms and partial matches
                    selected_words = set(symptom_lower.split())
                    disease_words = set(disease_symptom_lower.split())

                    # Check for partial matches and key terms
                    common_words = selected_words.intersection(disease_words)
                    if len(common_words) >= 1:  # At least one common word
                        matched_symptoms.append(selected_symptom)
                        score += 25  # Base score for symptom match
                        reasoning.append(f"✅ '{selected_symptom}' matches disease symptom '{disease_symptom}'")
                        print(f"DEBUG: Matched '{selected_symptom}' with '{disease_symptom}' (common words: {common_words})")
                        break

                    # Also check for substring matches (more flexible)
                    if symptom_lower in disease_symptom_lower or disease_symptom_lower in symptom_lower:
                        matched_symptoms.append(selected_symptom)
                        score += 20  # Slightly lower score for substring match
                        reasoning.append(f"✅ '{selected_symptom}' partially matches '{disease_symptom}'")
                        print(f"DEBUG: Substring match '{selected_symptom}' in '{disease_symptom}'")
                        break

                    # Also check for similar words (fuzzy matching) - improved
                    for selected_word in selected_words:
                        for disease_word in disease_words:
                            # More flexible fuzzy matching - check for similar words
                            if (len(selected_word) > 2 and len(disease_word) > 2 and
                                (selected_word in disease_word or disease_word in selected_word or
                                 selected_word[:-1] in disease_word or disease_word[:-1] in selected_word or
                                 selected_word[:3] == disease_word[:3] or  # First 3 letters match
                                 selected_word[-3:] == disease_word[-3:])):  # Last 3 letters match
                                matched_symptoms.append(selected_symptom)
                                score += 15  # Lower score for fuzzy matches
                                reasoning.append(f"✅ '{selected_symptom}' similar to disease symptom '{disease_symptom}'")
                                print(f"DEBUG: Fuzzy match '{selected_word}' ~ '{disease_word}'")
                                break

                    # Additional keyword matching for common symptom terms
                    symptom_keywords = {
                        'spots': ['spot', 'lesion', 'blotch', 'mark', 'blemish'],
                        'yellow': ['yellowing', 'chlorosis', 'pale', 'fading'],
                        'brown': ['browning', 'necrosis', 'burning', 'scorching'],
                        'wilting': ['wilt', 'droop', 'flaccid', 'limp'],
                        'curling': ['curl', 'crinkle', 'distort', 'deform'],
                        'powdery': ['powder', 'dust', 'coating', 'mildew'],
                        'mold': ['mildew', 'fungus', 'growth', 'mycelium'],
                        'halo': ['ring', 'circle', 'border', 'margin'],
                        'death': ['die', 'dead', 'dying', 'mortality']
                    }

                    # Check if selected symptom contains common keywords that match disease symptoms
                    for keyword, synonyms in symptom_keywords.items():
                        if keyword in symptom_lower:
                            for synonym in synonyms:
                                if synonym in disease_symptom_lower:
                                    matched_symptoms.append(selected_symptom)
                                    score += 18  # Good score for keyword-synonym matches
                                    reasoning.append(f"✅ '{selected_symptom}' (keyword: {keyword}) matches disease symptom '{disease_symptom}'")
                                    print(f"DEBUG: Keyword match '{keyword}' → '{synonym}' in '{disease_symptom}'")
                                    break

            # Boost score for multiple symptom matches
            if len(matched_symptoms) >= 2:
                score += len(matched_symptoms) * 8  # Bonus for multiple matches
                reasoning.append(f"✅ Multiple symptoms ({len(matched_symptoms)}) indicate {disease.replace('_', ' ').title()}")

            # Disease type relevance
            score += 15  # Base relevance for matching crop
            reasoning.append(f"✅ Disease commonly affects {crop_type.title()}")

            # Severity adjustment
            if severity == 'high':
                score *= 1.3
                reasoning.append("⚠️ High severity increases disease likelihood")
            elif severity == 'medium':
                score *= 1.1
                reasoning.append("⚡ Medium severity considered")

            # Image feature analysis (if available)
            if image_features is not None:
                # Enhanced image analysis based on disease visual cues
                color_variance = image_features[0][7]  # Approximate color variance

                # Check for disease-specific visual patterns
                if any(cue.lower() in ' '.join(data.get('visual_cues', [])).lower() for cue in ['spots', 'lesions', 'rings']):
                    if color_variance > 50:
                        score += 15
                        reasoning.append("🖼️ Image analysis supports spot/lesion pattern")

                if any(cue.lower() in ' '.join(data.get('visual_cues', [])).lower() for cue in ['powdery', 'mold', 'coating']):
                    if color_variance > 30 and color_variance < 60:
                        score += 15
                        reasoning.append("🖼️ Image analysis supports powdery/mold pattern")

            print(f"DEBUG: Disease {disease} score: {score}")

            # Minimum confidence threshold - lowered for better matching
            if score >= 5:  # Lowered threshold for better matching
                confidence = min(score, 95)
                matches.append({
                    'disease': disease,
                    'confidence': round(confidence, 1),
                    'details': data,
                    'method': 'Rule-based Analysis',
                    'reasoning': reasoning[:4],  # Show top 4 reasons
                    'matched_symptoms': matched_symptoms,
                    'type': data['type']
                })

        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        print(f"DEBUG: Found {len(matches)} matches")

        # Return top matches or default response
        if matches:
            return matches[:3]
        else:
            return [{
                'disease': 'Unable to diagnose',
                'confidence': 0,
                'details': {},
                'method': 'Rule-based',
                'reasoning': ['Insufficient symptom matches for confident diagnosis'],
                'matched_symptoms': [],
                'type': 'Unknown'
            }]

    def detect_disease(self, crop_type, image_path=None, symptoms=None, severity='medium'):
        """Main disease detection method - symptom-based with optional image support"""
        diseases = []

        # If we have an image, use it for additional confidence
        image_features = None
        if image_path:
            try:
                image_features = self.extract_image_features(image_path)
            except Exception as e:
                print(f"Image processing failed: {e}")
                image_features = None

        # Primary method: Enhanced rule-based approach using symptoms
        symptom_diseases = self.rule_based_detection(crop_type, symptoms or [], severity, image_features)

        return symptom_diseases

    def get_crop_specific_symptoms(self, crop_type):
        crop_specific_symptoms = {
            'leaf_symptoms': [],
            'plant_symptoms': [],
            'environmental_indicators': []
        }

        # Normalize crop type for matching
        crop_type_lower = crop_type.lower().strip()

        for disease, data in self.disease_data.items():
            # Check if this disease affects the selected crop
            affected_crops_lower = [c.lower().strip() for c in data['crop_affected']]

            # More flexible crop matching
            crop_matched = False
            for affected_crop in affected_crops_lower:
                if (affected_crop in crop_type_lower or
                    crop_type_lower in affected_crop or
                    affected_crop.replace(' ', '') in crop_type_lower.replace(' ', '') or
                    crop_type_lower.replace(' ', '') in affected_crop.replace(' ', '')):
                    crop_matched = True
                    break

            if crop_matched:
                # Add disease-specific symptoms to appropriate categories
                for symptom in data['symptoms']:
                    symptom_lower = symptom.lower()

                    # Categorize symptoms
                    if any(word in symptom_lower for word in ['leaf', 'leaves', 'spot', 'lesion', 'curl', 'yellow', 'brown', 'ring', 'halo', 'powdery', 'coating', 'mold']):
                        if symptom not in crop_specific_symptoms['leaf_symptoms']:
                            crop_specific_symptoms['leaf_symptoms'].append(symptom)
                    elif any(word in symptom_lower for word in ['plant', 'growth', 'stunted', 'wilting', 'death', 'drop', 'deformed', 'size']):
                        if symptom not in crop_specific_symptoms['plant_symptoms']:
                            crop_specific_symptoms['plant_symptoms'].append(symptom)
                    else:
                        # Environmental or general symptoms
                        if symptom not in crop_specific_symptoms['environmental_indicators']:
                            crop_specific_symptoms['environmental_indicators'].append(symptom)

        # Also add general symptoms that might be relevant
        general_leaf_symptoms = [
            'Yellowing of leaves', 'Brown spots on leaves', 'Leaf curling',
            'White powdery coating', 'Dark lesions', 'Premature leaf drop'
        ]

        general_plant_symptoms = [
            'Stunted growth', 'Wilting despite watering', 'Plant death',
            'Reduced yield', 'Flower drop', 'Fruit deformities'
        ]

        general_env_symptoms = [
            'Symptoms worse after rainfall', 'Symptoms appear in humid conditions',
            'Rapid symptom spread', 'Symptoms on lower leaves first'
        ]

        # Add general symptoms if they're not already covered by crop-specific ones
        for symptom in general_leaf_symptoms:
            if symptom not in crop_specific_symptoms['leaf_symptoms']:
                crop_specific_symptoms['leaf_symptoms'].append(symptom)

        for symptom in general_plant_symptoms:
            if symptom not in crop_specific_symptoms['plant_symptoms']:
                crop_specific_symptoms['plant_symptoms'].append(symptom)

        for symptom in general_env_symptoms:
            if symptom not in crop_specific_symptoms['environmental_indicators']:
                crop_specific_symptoms['environmental_indicators'].append(symptom)

        return crop_specific_symptoms

    def get_causal_analysis(self, disease, crop_type=None, severity='medium'):
        """Enhanced causal AI explanation for disease with visualizations"""
        if disease not in self.disease_data:
            return """
            Disease Analysis Not Available

            The specified disease is not in our database. Please consult with a local agricultural expert or extension service for proper diagnosis and treatment recommendations.

            Key points to consider:
            - Take clear photos of affected plant parts
            - Note environmental conditions (temperature, humidity, rainfall)
            - Consider recent weather patterns and irrigation practices
            - Check for insect activity or other stress factors
            """

        data = self.disease_data[disease]

        # Get pesticide recommendations if crop_type is provided
        pesticide_info = ""
        if crop_type and 'treatment' in data:
            treatment_data = data['treatment']

            # Determine severity level
            if severity == 'low':
                severity_key = 'low_severity'
            elif severity == 'medium':
                severity_key = 'medium_severity'
            else:  # high severity
                severity_key = 'high_severity'

            if severity_key in treatment_data and crop_type in treatment_data[severity_key]['pesticides']:
                pesticides = treatment_data[severity_key]['pesticides'][crop_type]
                frequency = treatment_data[severity_key]['frequency']

                pesticide_info = f"""
        💊 CROP-SPECIFIC PESTICIDE RECOMMENDATIONS:
        📅 Application Schedule: {frequency}

        Recommended Pesticides for {crop_type.title()}:
        {chr(10).join(f'• {pesticide}' for pesticide in pesticides)}

        ⚠️ SAFETY NOTES:
        • Always follow label instructions and local regulations
        • Apply during early morning or evening hours
        • Wear protective equipment (gloves, mask, protective clothing)
        • Avoid application during windy conditions
        • Keep children and pets away from treated areas
        """

        # Enhanced causal analysis with more detail
        analysis = f"""
        🌾 CAUSAL AI ANALYSIS: {disease.replace('_', ' ').title()}

        🔍 ROOT CAUSE ANALYSIS:
        {chr(10).join(f'• {cause}' for cause in data['causes'])}

        ⚠️ POTENTIAL EFFECTS IF UNTReATED:
        {chr(10).join(f'• {effect}' for effect in data['effects'])}

        🛡️ PREVENTION STRATEGIES:
        {chr(10).join(f'• {prevention}' for prevention in data['prevention'])}

        💊 GENERAL TREATMENT APPROACH:
        {pesticide_info if pesticide_info else "Consult agricultural experts for specific treatment recommendations based on your crop type and severity level."}

        📊 RISK ASSESSMENT DASHBOARD:
        ┌─────────────────────────────────────┐
        │ Disease Type:     {data['type']:<15} │
        │ Spread Rate:      {data['spread_rate']:<15} │
        │ Economic Impact:  {data['economic_impact']:<15} │
        │ Treatment Cost:   {data['treatment_cost']:<15} │
        └─────────────────────────────────────┘

        🎯 ACTION PLAN:
        1. Immediately isolate affected plants
        2. Apply recommended treatments within 24-48 hours
        3. Monitor nearby plants for early symptoms
        4. Implement prevention measures for next season
        5. Document conditions for future reference

        📞 PROFESSIONAL CONSULTATION:
        For severe outbreaks or uncertain diagnosis, consult:
        • Local agricultural extension service
        • Certified plant pathologist
        • Agricultural university specialists

        📈 MONITORING CHECKLIST:
        □ Daily inspection of affected area
        □ Track symptom progression
        □ Monitor weather conditions
        □ Document treatment effectiveness
        □ Plan preventive measures for next season
        """

        return analysis.strip()

    def get_pesticide_recommendations(self, disease, crop_type, severity):
        """Get crop-specific and severity-based pesticide recommendations"""
        if disease not in self.disease_data or 'treatment' not in self.disease_data[disease]:
            return None

        treatment_data = self.disease_data[disease]['treatment']

        # Determine severity level
        if severity == 'low':
            severity_key = 'low_severity'
        elif severity == 'medium':
            severity_key = 'medium_severity'
        else:  # high severity
            severity_key = 'high_severity'

        if severity_key in treatment_data and crop_type in treatment_data[severity_key]['pesticides']:
            return {
                'frequency': treatment_data[severity_key]['frequency'],
                'pesticides': treatment_data[severity_key]['pesticides'][crop_type]
            }

        return None

    def simple_detect_disease(self, crop, symptoms):
        """Simple rule-based disease detection"""
        detected = []
        for rule in self.disease_rules:
            if rule["crop"] == crop:
                match_count = len(set(symptoms) & set(rule["symptoms"]))
                if match_count > 0:
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": list(set(symptoms) & set(rule["symptoms"]))
                    })
        return detected

    def simple_causal_analysis(self, detected):
        """Simple causal analysis"""
        causal_info = []
        for r in detected:
            info = f"Disease {r['disease']} triggered because symptoms {r['matched_symptoms']} matched known patterns."
            causal_info.append(info)
        return causal_info

# Usage example
if __name__ == "__main__":
    # Test crop recommendation
    recommender = CropRecommender()
    soil_data = {'N': 75, 'P': 35, 'K': 35, 'ph': 6.8, 'temperature': 24, 'humidity': 75, 'rainfall': 120}
    recommendations = recommender.recommend_crops(soil_data, "India")

    print("Crop Recommendations:")
    for rec in recommendations:
        print(f"- {rec['crop'].title()}: {rec['confidence']}% confidence ({rec.get('method', 'Rule-based')})")

    # Test disease detection
    detector = DiseaseDetector()
    results = detector.simple_detect_disease("Tomato", ["yellow leaves", "spots on leaves"])

    print("\nDisease Detection Results:")
    for r in results:
        print(f"Crop: Tomato")
        print(f"Disease Detected: {r['disease']}")
        print(f"Pesticide Recommended: {r['pesticide']}")
        print(f"Cost per hectare: ${r['cost_per_ha']}")
        print(f"Matched Symptoms: {r['matched_symptoms']}")

    causal_explanations = detector.simple_causal_analysis(results)
    print("\nCausal Explanations:")
    for e in causal_explanations:
        print("-", e)
    def __init__(self):
        # Sample rule-based crop disease database
        self.disease_rules = [
            {
                "crop": "Tomato",
                "symptoms": ["yellow leaves", "spots on leaves", "wilting"],
                "disease": "Early Blight",
                "pesticide": "Chlorothalonil",
                "cost_per_ha": 120
            },
            {
                "crop": "Wheat",
                "symptoms": ["yellow streaks", "leaf rust"],
                "disease": "Leaf Rust",
                "pesticide": "Propiconazole",
                "cost_per_ha": 90
            },
            {
                "crop": "Rice",
                "symptoms": ["brown spots", "stunted growth"],
                "disease": "Brown Spot",
                "pesticide": "Mancozeb",
                "cost_per_ha": 100
            }
        ]

        self.disease_data = {
            'early_blight': {
                'type': 'Fungal',
                'crop_affected': ['tomato', 'potato', 'pepper', 'eggplant'],
                'symptoms': ['dark spots with concentric rings', 'yellow halos around spots', 'leaf curling', 'brown lesions', 'target-like spots on leaves', 'spots starting from lower leaves', 'irregular shaped lesions'],
                'causes': ['Alternaria solani fungus', 'warm humid conditions (20-30°C)', 'poor air circulation', 'overcrowded plants', 'prolonged leaf wetness'],
                'effects': ['reduced photosynthesis by 50-70%', 'premature leaf drop', 'reduced yield by 20-50%', 'plant death in severe cases', 'fruit quality deterioration'],
                'prevention': ['crop rotation every 2-3 years', 'proper plant spacing (45-60cm)', 'fungicide application before symptoms', 'remove and destroy infected leaves immediately', 'avoid overhead watering', 'use drip irrigation'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ],
                            'potato': [
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'},
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'}
                            ],
                            'pepper': [
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ],
                            'eggplant': [
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'},
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'potato': [
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'},
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'}
                            ],
                            'pepper': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'eggplant': [
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'},
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'potato': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ],
                            'pepper': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'eggplant': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['spot size > 1cm diameter', 'multiple spots per leaf (>5)', 'spots on upper leaves', 'rapid spot expansion', 'defoliation > 25%'],
                'visual_cues': ['concentric ring pattern', 'yellow halo around dark center', 'irregular lesion shape'],
                'spread_rate': 'Moderate to High',
                'economic_impact': 'High (20-50% yield loss)',
                'treatment_cost': 'Medium ($50-150/ha)'
            },
            'late_blight': {
                'type': 'Fungal',
                'crop_affected': ['tomato', 'potato'],
                'symptoms': ['water-soaked lesions on leaves', 'white fungal growth on leaf undersides', 'rapid plant death within days', 'dark brown spots with greasy appearance', 'rapid defoliation', 'stem lesions', 'fruit rot'],
                'causes': ['Phytophthora infestans fungus', 'cool moist conditions (15-25°C)', 'high humidity (>80%)', 'poor drainage', 'prolonged leaf wetness (>6 hours)', 'infected seed tubers'],
                'effects': ['complete crop loss within 2-3 days under favorable conditions', 'rapid spore spread to nearby plants', 'contamination of tubers and soil', 'economic loss up to 100%', 'secondary bacterial infections'],
                'prevention': ['use resistant varieties (ex: Kufri Jyoti, Kufri Himalini)', 'proper irrigation management', 'good field sanitation', 'avoid overhead watering', 'maintain proper plant spacing', 'apply preventive fungicides', 'early planting to avoid peak humidity'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 7-10 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'},
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'}
                            ],
                            'potato': [
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 5-7 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'potato': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Dimethomorph 50% WP', 'dosage': '1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹1200/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Emergency treatment every 3-5 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Fluopicolide 4.44% + Fosetyl-Al 66.7% WG', 'dosage': '2.5g/L', 'price_per_kg': 1500, 'estimated_cost': '₹3750/ha'}
                            ],
                            'potato': [
                                {'name': 'Fluazinam 50% WP', 'dosage': '1g/L', 'price_per_kg': 1800, 'estimated_cost': '₹1800/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 850, 'estimated_cost': '₹2125/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['rapid spread rate', 'white mold on leaf undersides', 'plant collapse within 48 hours', 'stem infection', 'multiple plants affected'],
                'visual_cues': ['white cottony growth on leaf undersides', 'water-soaked appearance', 'rapid lesion expansion'],
                'spread_rate': 'Very High (explosive outbreaks)',
                'economic_impact': 'Severe (50-100% crop loss)',
                'treatment_cost': 'High ($100-300/ha)'
            },
            'powdery_mildew': {
                'type': 'Fungal',
                'crop_affected': ['cucumber', 'melon', 'pumpkin', 'grape', 'rose', 'pea'],
                'symptoms': ['white powdery coating on leaf surfaces', 'yellowing of infected leaves', 'stunted growth', 'distorted leaves and shoots', 'premature leaf drop', 'reduced fruit quality', 'white patches that can be rubbed off'],
                'causes': ['Erysiphe cichoracearum and other Erysiphe species', 'high humidity (60-80%) with moderate temperatures (20-25°C)', 'poor air circulation', 'nitrogen deficiency', 'dense plant canopy', 'overhead irrigation'],
                'effects': ['reduced photosynthesis by 30-50%', 'lower quality produce', 'yield loss up to 30%', 'reduced plant vigor', 'increased susceptibility to other diseases', 'premature plant death'],
                'prevention': ['proper plant spacing for air circulation', 'avoid overhead irrigation', 'resistant varieties selection', 'balanced fertilization (avoid excess nitrogen)', 'good sanitation practices', 'remove infected plant debris', 'improve ventilation'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'cucumber': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Potassium bicarbonate', 'dosage': '5g/L', 'price_per_kg': 200, 'estimated_cost': '₹1000/ha'}
                            ],
                            'melon': [
                                {'name': 'Wettable sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Neem oil 1%', 'dosage': '5ml/L', 'price_per_liter': 350, 'estimated_cost': '₹1750/ha'}
                            ],
                            'pumpkin': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Potassium bicarbonate', 'dosage': '5g/L', 'price_per_kg': 200, 'estimated_cost': '₹1000/ha'}
                            ],
                            'grape': [
                                {'name': 'Sulfur 80% WP', 'dosage': '2.5g/L', 'price_per_kg': 120, 'estimated_cost': '₹300/ha'},
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'}
                            ],
                            'rose': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ],
                            'pea': [
                                {'name': 'Sulfur 80% WP', 'dosage': '3g/L', 'price_per_kg': 120, 'estimated_cost': '₹360/ha'},
                                {'name': 'Potassium bicarbonate', 'dosage': '5g/L', 'price_per_kg': 200, 'estimated_cost': '₹1000/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'cucumber': [
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ],
                            'melon': [
                                {'name': 'Hexaconazole 5% EC', 'dosage': '1ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹2200/ha'},
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'}
                            ],
                            'pumpkin': [
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ],
                            'grape': [
                                {'name': 'Flusilazole 40% EC', 'dosage': '0.25ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹800/ha'},
                                {'name': 'Penconazole 10% EC', 'dosage': '0.5ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹1400/ha'}
                            ],
                            'rose': [
                                {'name': 'Hexaconazole 5% EC', 'dosage': '1ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹2200/ha'},
                                {'name': 'Propiconazole 25% EC', 'dosage': '1ml/L', 'price_per_liter': 2400, 'estimated_cost': '₹2400/ha'}
                            ],
                            'pea': [
                                {'name': 'Myclobutanil 10% WP', 'dosage': '0.4g/L', 'price_per_kg': 1800, 'estimated_cost': '₹720/ha'},
                                {'name': 'Triadimefon 25% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'cucumber': [
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'melon': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Kresoxim-methyl 44.3% SC', 'dosage': '0.5ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹1750/ha'}
                            ],
                            'pumpkin': [
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ],
                            'grape': [
                                {'name': 'Trifloxystrobin 25% + Tebuconazole 50% WG', 'dosage': '0.5g/L', 'price_per_kg': 2200, 'estimated_cost': '₹1100/ha'},
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'}
                            ],
                            'rose': [
                                {'name': 'Azoxystrobin 23% SC', 'dosage': '1ml/L', 'price_per_liter': 2800, 'estimated_cost': '₹2800/ha'},
                                {'name': 'Kresoxim-methyl 44.3% SC', 'dosage': '0.5ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹1750/ha'}
                            ],
                            'pea': [
                                {'name': 'Tebuconazole 25.9% EC', 'dosage': '1ml/L', 'price_per_liter': 2600, 'estimated_cost': '₹2600/ha'},
                                {'name': 'Difenoconazole 25% EC', 'dosage': '0.5ml/L', 'price_per_liter': 3200, 'estimated_cost': '₹1600/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['powdery coating density (>50% leaf coverage)', 'leaf yellowing extent', 'growth stunting (>20%)', 'multiple plant infection', 'early season occurrence'],
                'visual_cues': ['white powdery coating', 'can be easily rubbed off', 'starts as small white spots'],
                'spread_rate': 'High (wind and water dispersed)',
                'economic_impact': 'Medium (15-30% yield loss)',
                'treatment_cost': 'Low to Medium ($30-80/ha)'
            },
            'bacterial_spot': {
                'type': 'Bacterial',
                'crop_affected': ['tomato', 'pepper'],
                'symptoms': ['small water-soaked spots on leaves', 'yellow halos around spots', 'leaf perforation as spots age', 'dark raised lesions on fruit', 'severe defoliation in humid conditions', 'angular leaf spots', 'spots turn brown and necrotic'],
                'causes': ['Xanthomonas vesicatoria and related species', 'warm wet conditions (25-30°C)', 'infected seeds or transplants', 'splashing water from rain or irrigation', 'high humidity (>85%)', 'wounds from insects or hail'],
                'effects': ['severe defoliation in humid conditions', 'reduced fruit quality and market value', 'yield loss up to 40%', 'secondary infections by other pathogens', 'unmarketable fruit due to spotting'],
                'prevention': ['use disease-free seeds and transplants', 'crop rotation with non-host plants (3 years)', 'avoid overhead watering', 'copper-based preventive sprays', 'good field sanitation', 'remove volunteer plants', 'use resistant varieties'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ],
                            'pepper': [
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Streptomycin sulfate 90% SP', 'dosage': '0.1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹120/ha'},
                                {'name': 'Kasugamycin 3% SL', 'dosage': '2ml/L', 'price_per_liter': 1800, 'estimated_cost': '₹3600/ha'}
                            ],
                            'pepper': [
                                {'name': 'Streptomycin sulfate 90% SP', 'dosage': '0.1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹120/ha'},
                                {'name': 'Kasugamycin 3% SL', 'dosage': '2ml/L', 'price_per_liter': 1800, 'estimated_cost': '₹3600/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'tomato': [
                                {'name': 'Oxytetracycline hydrochloride 50% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'},
                                {'name': 'Validamycin 3% L', 'dosage': '2ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹4400/ha'}
                            ],
                            'pepper': [
                                {'name': 'Oxytetracycline hydrochloride 50% WP', 'dosage': '0.5g/L', 'price_per_kg': 950, 'estimated_cost': '₹475/ha'},
                                {'name': 'Validamycin 3% L', 'dosage': '2ml/L', 'price_per_liter': 2200, 'estimated_cost': '₹4400/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['halo formation around spots', 'rapid spot expansion (>1mm/day)', 'leaf perforation', 'fruit infection', 'multiple plants affected'],
                'visual_cues': ['angular spots (limited by leaf veins)', 'water-soaked appearance initially', 'yellow halos'],
                'spread_rate': 'High (rain splash dispersed)',
                'economic_impact': 'High (30-50% yield loss)',
                'treatment_cost': 'Medium ($60-120/ha)'
            },
            'bacterial_wilt': {
                'type': 'Bacterial',
                'crop_affected': ['tomato', 'pepper', 'potato', 'eggplant'],
                'symptoms': ['sudden wilting of entire plant', 'yellowing and browning of leaves', 'vascular discoloration (brown streaks)', 'plant death within days', 'no response to watering', 'stunted growth', 'adventitious roots on stems'],
                'causes': ['Ralstonia solanacearum bacteria', 'warm soil temperatures (25-35°C)', 'high soil moisture', 'poor soil drainage', 'infected soil or water', 'mechanical damage to roots', 'monoculture practices'],
                'effects': ['complete plant death', 'soil contamination for multiple seasons', 'yield loss up to 80%', 'reduced crop quality', 'economic losses from replanting', 'spread to adjacent fields'],
                'prevention': ['use resistant varieties', 'crop rotation with non-solanaceous crops', 'soil solarization', 'improve soil drainage', 'use clean irrigation water', 'avoid root damage during cultivation', 'soil testing for pathogen presence'],
                'treatment': ['no effective chemical control available', 'remove infected plants immediately with soil', 'soil fumigation for future crops', 'improve soil drainage', 'use resistant varieties', 'avoid susceptible crops for 2-3 years'],
                'severity_indicators': ['vascular browning in stems', 'rapid wilting despite adequate moisture', 'plant collapse within 3-5 days', 'multiple plants affected suddenly'],
                'visual_cues': ['brown vascular discoloration', 'hollow stem appearance', 'bacterial ooze from cut stems'],
                'spread_rate': 'Very High (soil and water borne)',
                'economic_impact': 'Severe (50-90% crop loss)',
                'treatment_cost': 'High (soil remediation costs)'
            },

            # Viral Diseases
            'tomato_yellow_leaf_curl': {
                'type': 'Viral',
                'crop_affected': ['tomato'],
                'symptoms': ['upward curling of leaves', 'yellowing between veins', 'stunted plant growth', 'reduced leaf size', 'flower drop', 'deformed fruit', 'purple tinged leaves', 'shortened internodes'],
                'causes': ['Tomato yellow leaf curl virus (TYLCV)', 'whitefly vector (Bemisia tabaci)', 'infected transplants or seeds', 'warm temperatures favoring whitefly reproduction', 'monoculture tomato production', 'poor vector control'],
                'effects': ['stunted plant growth and reduced vigor', 'significant yield reduction (30-70%)', 'poor fruit quality and size', 'delayed maturity', 'increased susceptibility to other stresses', 'economic losses from reduced marketable yield'],
                'prevention': ['use virus-free transplants and seeds', 'whitefly vector control with insecticides', 'use reflective mulches to repel whiteflies', 'remove infected plants immediately', 'crop rotation with non-host crops', 'use resistant varieties', 'early planting to avoid peak whitefly season'],
                'treatment': ['no direct chemical treatment for virus', 'control whitefly vectors with systemic insecticides', 'remove infected plants immediately', 'use insect-proof netting', 'apply mineral oils to disrupt virus transmission', 'maintain field sanitation'],
                'severity_indicators': ['leaf curling >50% of plant', 'stunted growth (<50% normal height)', 'yellow vein clearing', 'multiple plants infected', 'early season infection'],
                'visual_cues': ['severe upward leaf curl', 'yellowing between leaf veins', 'stunted bushy appearance'],
                'spread_rate': 'High (vector transmitted)',
                'economic_impact': 'High (40-70% yield loss)',
                'treatment_cost': 'Medium ($80-150/ha)'
            },

            # Nutrient Deficiencies (often misdiagnosed as diseases)
            'nitrogen_deficiency': {
                'type': 'Nutritional',
                'crop_affected': ['all crops'],
                'symptoms': ['older leaves turn yellow first', 'general yellowing of foliage', 'stunted growth', 'thin stems', 'reduced tillering', 'small leaves', 'premature flowering', 'pale green to yellow coloration'],
                'causes': ['insufficient nitrogen in soil', 'poor soil fertility', 'excessive leaching due to heavy rainfall', 'high soil pH reducing availability', 'crop rotation without nitrogen fixation', 'previous crop nitrogen depletion'],
                'effects': ['reduced photosynthesis and plant vigor', 'lower protein content in grains', 'yield reduction up to 30%', 'increased susceptibility to diseases', 'poor root development', 'delayed maturity'],
                'prevention': ['soil testing before planting', 'proper fertilization with nitrogen sources', 'use of organic matter and compost', 'crop rotation with legumes', 'split application of nitrogen fertilizer', 'avoid over-irrigation that causes leaching'],
                'treatment': ['apply nitrogen fertilizers (urea, ammonium nitrate)', 'foliar application of urea (1-2%)', 'incorporate organic matter', 'use slow-release nitrogen sources', 'ensure proper irrigation', 'adjust soil pH if necessary'],
                'severity_indicators': ['yellowing starting from lower leaves', 'general plant weakness', 'small leaf size', 'reduced growth rate', 'multiple plants affected uniformly'],
                'visual_cues': ['uniform yellowing pattern', 'starts from older leaves', 'no spots or lesions'],
                'spread_rate': 'Uniform across field',
                'economic_impact': 'Medium (15-30% yield loss)',
                'treatment_cost': 'Low ($20-50/ha)'
            },
            'potassium_deficiency': {
                'type': 'Nutritional',
                'crop_affected': ['all crops'],
                'symptoms': ['yellowing and browning of leaf margins', 'scorching of older leaves', 'weak stems', 'small fruit size', 'poor fruit quality', 'leaf curling', 'reduced disease resistance', 'brown spots on leaves'],
                'causes': ['low potassium levels in soil', 'sandy soils with poor cation exchange', 'excessive calcium or magnesium', 'high rainfall causing leaching', 'continuous cropping without replacement', 'acidic soil conditions'],
                'effects': ['reduced plant vigor and stress tolerance', 'poor fruit development and quality', 'yield reduction up to 25%', 'increased disease susceptibility', 'reduced winter hardiness', 'poor root development'],
                'prevention': ['regular soil testing for potassium levels', 'use of potassium-rich fertilizers', 'maintain proper soil pH (6.0-7.0)', 'avoid excessive nitrogen application', 'use organic matter to improve CEC', 'balanced fertilization program'],
                'treatment': ['apply potassium fertilizers (potassium chloride, sulfate)', 'foliar application of potassium nitrate', 'use potassium-rich organic sources', 'improve soil structure with organic matter', 'ensure proper irrigation practices', 'avoid soil compaction'],
                'severity_indicators': ['marginal leaf burn', 'scorching of older leaves', 'weak stem strength', 'poor fruit development', 'uniform field pattern'],
                'visual_cues': ['leaf margin necrosis', 'brown scorching starting from edges', 'interveinal chlorosis'],
                'spread_rate': 'Gradual across field',
                'economic_impact': 'Medium (15-25% yield loss)',
                'treatment_cost': 'Low to Medium ($30-70/ha)'
            },

            # Insect-Related Issues
            'onion_downy_mildew': {
                'type': 'Fungal',
                'crop_affected': ['onion', 'garlic', 'shallot'],
                'symptoms': ['white powdery coating on leaves', 'reduced yield', 'symptoms on lower leaves first', 'yellowing of leaves', 'stunted growth', 'premature leaf dieback'],
                'causes': ['Peronospora destructor fungus', 'cool moist conditions (15-20°C)', 'high humidity (>80%)', 'poor air circulation', 'dense plantings', 'infected seeds or soil'],
                'effects': ['reduced photosynthesis leading to lower bulb size', 'yield loss up to 50%', 'poor bulb quality', 'increased susceptibility to secondary infections', 'complete crop failure in severe cases'],
                'prevention': ['use disease-free seeds', 'crop rotation every 3-4 years', 'proper plant spacing (10-15cm)', 'improve air circulation', 'avoid overhead irrigation', 'remove crop debris after harvest'],
                'treatment': {
                    'low_severity': {
                        'frequency': 'Preventive application every 10-14 days',
                        'pesticides': {
                            'onion': [
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'},
                                {'name': 'Copper oxychloride 50% WP', 'dosage': '3g/L', 'price_per_kg': 450, 'estimated_cost': '₹1350/ha'}
                            ],
                            'garlic': [
                                {'name': 'Chlorothalonil 75% WP', 'dosage': '2g/L', 'price_per_kg': 520, 'estimated_cost': '₹1040/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ],
                            'shallot': [
                                {'name': 'Copper hydroxide 77% WP', 'dosage': '2.5g/L', 'price_per_kg': 680, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Mancozeb 75% WP', 'dosage': '2.5g/L', 'price_per_kg': 380, 'estimated_cost': '₹950/ha'}
                            ]
                        }
                    },
                    'medium_severity': {
                        'frequency': 'Curative application every 7-10 days',
                        'pesticides': {
                            'onion': [
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'},
                                {'name': 'Metalaxyl 8% + Mancozeb 64% WP', 'dosage': '2.5g/L', 'price_per_kg': 780, 'estimated_cost': '₹1950/ha'}
                            ],
                            'garlic': [
                                {'name': 'Dimethomorph 50% WP', 'dosage': '1g/L', 'price_per_kg': 1200, 'estimated_cost': '₹1200/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ],
                            'shallot': [
                                {'name': 'Metalaxyl-M 4% + Mancozeb 68% WG', 'dosage': '2.5g/L', 'price_per_kg': 920, 'estimated_cost': '₹2300/ha'},
                                {'name': 'Cymoxanil 8% + Mancozeb 64% WP', 'dosage': '2g/L', 'price_per_kg': 850, 'estimated_cost': '₹1700/ha'}
                            ]
                        }
                    },
                    'high_severity': {
                        'frequency': 'Intensive treatment every 5-7 days',
                        'pesticides': {
                            'onion': [
                                {'name': 'Fluopicolide 4.44% + Fosetyl-Al 66.7% WG', 'dosage': '2.5g/L', 'price_per_kg': 1500, 'estimated_cost': '₹3750/ha'},
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'}
                            ],
                            'garlic': [
                                {'name': 'Fluazinam 50% WP', 'dosage': '1g/L', 'price_per_kg': 1800, 'estimated_cost': '₹1800/ha'},
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'}
                            ],
                            'shallot': [
                                {'name': 'Fluopicolide 4.44% + Fosetyl-Al 66.7% WG', 'dosage': '2.5g/L', 'price_per_kg': 1500, 'estimated_cost': '₹3750/ha'},
                                {'name': 'Mandipropamid 23.4% SC', 'dosage': '0.8ml/L', 'price_per_liter': 3500, 'estimated_cost': '₹2800/ha'}
                            ]
                        }
                    }
                },
                'severity_indicators': ['white coating density (>50% leaf coverage)', 'yellowing extent on lower leaves', 'stunted growth (>20%)', 'rapid symptom progression', 'multiple plants affected'],
                'visual_cues': ['white powdery coating on leaf surfaces', 'starts on lower leaves', 'yellowing between veins'],
                'spread_rate': 'High (wind and water dispersed)',
                'economic_impact': 'High (30-50% yield loss)',
                'treatment_cost': 'Medium ($100-200/ha)'
            },
        }

        # Symptom categories for dropdown selection - aligned with disease database
        self.symptom_categories = {
            'leaf_symptoms': [
                'Dark spots with concentric rings',  # Early Blight
                'Yellow halos around spots',        # Early Blight
                'Leaf curling',                     # Early Blight, Viral
                'Brown lesions',                    # Early Blight
                'Target-like spots on leaves',      # Early Blight
                'White powdery coating on leaves',  # Powdery Mildew, Onion Downy Mildew
                'Water-soaked lesions',             # Late Blight, Bacterial Spot
                'Angular leaf spots',               # Bacterial Spot
                'Leaf curling upward',              # Viral diseases
                'Yellowing between leaf veins',     # Viral diseases, Nutrient deficiency
                'Yellowing starting from leaf margins', # Potassium deficiency
                'Premature leaf drop',              # Multiple diseases
                'Stunted leaf growth',              # Multiple diseases
                'White mold on leaf undersides',    # Late Blight
                'Greasy appearance on leaves',      # Late Blight
                'Symptoms on lower leaves first'    # Onion Downy Mildew
            ],
            'plant_symptoms': [
                'Sudden wilting of entire plant',   # Bacterial Wilt
                'Stunted plant growth',             # Viral, Nutrient deficiencies
                'Plant death within days',          # Late Blight, Bacterial Wilt
                'Flower drop',                      # Viral diseases
                'Deformed fruit',                   # Viral diseases, Bacterial Spot
                'Small fruit size',                 # Nutrient deficiencies
                'Purple tinged leaves',             # Viral diseases
                'No response to watering',          # Bacterial Wilt
                'Reduced yield'                     # Onion Downy Mildew
            ],
            'environmental_indicators': [
                'Symptoms worse during humid periods',
                'Symptoms appear after rainfall',
                'Symptoms worse in warm conditions',
                'Symptoms worse in cool conditions',
                'Rapid symptom progression',
                'Symptoms spreading to nearby plants',
                'Symptoms on lower leaves first',
                'Symptoms on upper leaves first'
            ]
        }

        self.load_or_train_model()

    def load_or_train_model(self):
        """Load or train the disease detection model"""
        try:
            import joblib
            self.model = joblib.load('models/disease_detection_model.pkl')
            self.label_encoder = joblib.load('models/disease_label_encoder.pkl')
        except (ImportError, FileNotFoundError, IOError):
            # joblib not available or model files don't exist, use rule-based only
            # In a real implementation, you would train a CNN model here
            self.model = None
            self.label_encoder = None

    def extract_image_features(self, image_path):
        """Extract features from crop leaf image"""
        try:
            from PIL import Image
            import cv2
            import numpy as np

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize for consistent processing
            img_resized = cv2.resize(img_rgb, (224, 224))

            # Extract color features (mean and std of each channel)
            features = []
            for channel in range(3):  # R, G, B
                channel_data = img_resized[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data)
                ])

            # Extract texture features using gray scale
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

            # Calculate texture features
            features.extend([
                np.mean(gray),  # Mean intensity
                np.std(gray),   # Standard deviation
                np.var(gray)    # Variance
            ])

            # Calculate edges using Sobel
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Edge magnitude
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features.extend([
                np.mean(edge_magnitude),
                np.std(edge_magnitude)
            ])

            # HSV color space features
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
            for channel in range(3):  # H, S, V
                channel_data = hsv[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data)
                ])

            return np.array(features).reshape(1, -1)

        except ImportError:
            print("Warning: Image processing libraries (PIL, cv2) not available")
            return None
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def rule_based_detection(self, crop_type, selected_symptoms, severity, image_features=None):
        """Enhanced rule-based disease detection with comprehensive symptom matching"""
        matches = []

        # Debug: Print what symptoms we're receiving
        print(f"DEBUG: Crop type: {crop_type}")
        print(f"DEBUG: Selected symptoms: {selected_symptoms}")
        print(f"DEBUG: Severity: {severity}")

        # Normalize crop type
        crop_type_lower = crop_type.lower().strip() if crop_type else ""

        for disease, data in self.disease_data.items():
            score = 0
            reasoning = []
            matched_symptoms = []

            # Check if crop is affected by this disease (case-insensitive and flexible matching)
            affected_crops_lower = [c.lower().strip() for c in data['crop_affected']]

            # More flexible crop matching
            crop_matched = False
            for affected_crop in affected_crops_lower:
                if (affected_crop in crop_type_lower or
                    crop_type_lower in affected_crop or
                    affected_crop.replace(' ', '') in crop_type_lower.replace(' ', '') or
                    crop_type_lower.replace(' ', '') in affected_crop.replace(' ', '')):
                    crop_matched = True
                    break

            if not crop_matched:
                print(f"DEBUG: Skipping {disease} - not for crop {crop_type}")
                continue

            print(f"DEBUG: Checking disease: {disease}")

            # If no symptoms provided, give basic matching score
            if not selected_symptoms or len(selected_symptoms) == 0:
                score += 15  # Base score for crop match
                reasoning.append(f"✅ Disease commonly affects {crop_type.title()}")
                matches.append({
                    'disease': disease,
                    'confidence': min(score, 95),
                    'details': data,
                    'method': 'Rule-based Analysis',
                    'reasoning': reasoning[:2],
                    'matched_symptoms': [],
                    'type': data['type']
                })
                continue

            # Symptom matching with categories - more flexible matching
            for selected_symptom in selected_symptoms:
                symptom_lower = selected_symptom.lower().strip()
                print(f"DEBUG: Checking symptom: '{selected_symptom}'")

                # Check against disease symptoms with flexible matching
                for disease_symptom in data['symptoms']:
                    disease_symptom_lower = disease_symptom.lower()
                    print(f"DEBUG: Comparing with disease symptom: '{disease_symptom}'")

                    # More flexible matching - check for key terms and partial matches
                    selected_words = set(symptom_lower.split())
                    disease_words = set(disease_symptom_lower.split())

                    # Check for partial matches and key terms
                    common_words = selected_words.intersection(disease_words)
                    if len(common_words) >= 1:  # At least one common word
                        matched_symptoms.append(selected_symptom)
                        score += 25  # Base score for symptom match
                        reasoning.append(f"✅ '{selected_symptom}' matches disease symptom '{disease_symptom}'")
                        print(f"DEBUG: Matched '{selected_symptom}' with '{disease_symptom}' (common words: {common_words})")
                        break

                    # Also check for substring matches (more flexible)
                    if symptom_lower in disease_symptom_lower or disease_symptom_lower in symptom_lower:
                        matched_symptoms.append(selected_symptom)
                        score += 20  # Slightly lower score for substring match
                        reasoning.append(f"✅ '{selected_symptom}' partially matches '{disease_symptom}'")
                        print(f"DEBUG: Substring match '{selected_symptom}' in '{disease_symptom}'")
                        break

                    # Also check for similar words (fuzzy matching) - improved
                    for selected_word in selected_words:
                        for disease_word in disease_words:
                            # More flexible fuzzy matching - check for similar words
                            if (len(selected_word) > 2 and len(disease_word) > 2 and
                                (selected_word in disease_word or disease_word in selected_word or
                                 selected_word[:-1] in disease_word or disease_word[:-1] in selected_word or
                                 selected_word[:3] == disease_word[:3] or  # First 3 letters match
                                 selected_word[-3:] == disease_word[-3:])):  # Last 3 letters match
                                matched_symptoms.append(selected_symptom)
                                score += 15  # Lower score for fuzzy matches
                                reasoning.append(f"✅ '{selected_symptom}' similar to disease symptom '{disease_symptom}'")
                                print(f"DEBUG: Fuzzy match '{selected_word}' ~ '{disease_word}'")
                                break

                    # Additional keyword matching for common symptom terms
                    symptom_keywords = {
                        'spots': ['spot', 'lesion', 'blotch', 'mark', 'blemish'],
                        'yellow': ['yellowing', 'chlorosis', 'pale', 'fading'],
                        'brown': ['browning', 'necrosis', 'burning', 'scorching'],
                        'wilting': ['wilt', 'droop', 'flaccid', 'limp'],
                        'curling': ['curl', 'crinkle', 'distort', 'deform'],
                        'powdery': ['powder', 'dust', 'coating', 'mildew'],
                        'mold': ['mildew', 'fungus', 'growth', 'mycelium'],
                        'halo': ['ring', 'circle', 'border', 'margin'],
                        'death': ['die', 'dead', 'dying', 'mortality']
                    }

                    # Check if selected symptom contains common keywords that match disease symptoms
                    for keyword, synonyms in symptom_keywords.items():
                        if keyword in symptom_lower:
                            for synonym in synonyms:
                                if synonym in disease_symptom_lower:
                                    matched_symptoms.append(selected_symptom)
                                    score += 18  # Good score for keyword-synonym matches
                                    reasoning.append(f"✅ '{selected_symptom}' (keyword: {keyword}) matches disease symptom '{disease_symptom}'")
                                    print(f"DEBUG: Keyword match '{keyword}' → '{synonym}' in '{disease_symptom}'")
                                    break

            # Boost score for multiple symptom matches
            if len(matched_symptoms) >= 2:
                score += len(matched_symptoms) * 8  # Bonus for multiple matches
                reasoning.append(f"✅ Multiple symptoms ({len(matched_symptoms)}) indicate {disease.replace('_', ' ').title()}")

            # Disease type relevance
            score += 15  # Base relevance for matching crop
            reasoning.append(f"✅ Disease commonly affects {crop_type.title()}")

            # Severity adjustment
            if severity == 'high':
                score *= 1.3
                reasoning.append("⚠️ High severity increases disease likelihood")
            elif severity == 'medium':
                score *= 1.1
                reasoning.append("⚡ Medium severity considered")

            # Image feature analysis (if available)
            if image_features is not None:
                # Enhanced image analysis based on disease visual cues
                color_variance = image_features[0][7]  # Approximate color variance

                # Check for disease-specific visual patterns
                if any(cue.lower() in ' '.join(data.get('visual_cues', [])).lower() for cue in ['spots', 'lesions', 'rings']):
                    if color_variance > 50:
                        score += 15
                        reasoning.append("🖼️ Image analysis supports spot/lesion pattern")

                if any(cue.lower() in ' '.join(data.get('visual_cues', [])).lower() for cue in ['powdery', 'mold', 'coating']):
                    if color_variance > 30 and color_variance < 60:
                        score += 15
                        reasoning.append("🖼️ Image analysis supports powdery/mold pattern")

            print(f"DEBUG: Disease {disease} score: {score}")

            # Minimum confidence threshold - lowered for better matching
            if score >= 5:  # Lowered threshold for better matching
                confidence = min(score, 95)
                matches.append({
                    'disease': disease,
                    'confidence': round(confidence, 1),
                    'details': data,
                    'method': 'Rule-based Analysis',
                    'reasoning': reasoning[:4],  # Show top 4 reasons
                    'matched_symptoms': matched_symptoms,
                    'type': data['type']
                })

        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        print(f"DEBUG: Found {len(matches)} matches")

        # Return top matches or default response
        if matches:
            return matches[:3]
        else:
            return [{
                'disease': 'Unable to diagnose',
                'confidence': 0,
                'details': {},
                'method': 'Rule-based',
                'reasoning': ['Insufficient symptom matches for confident diagnosis'],
                'matched_symptoms': [],
                'type': 'Unknown'
            }]

    def detect_disease(self, crop_type, image_path=None, symptoms=None, severity='medium'):
        """Main disease detection method - model-based with image, fallback to rule-based"""
        diseases = []

        # If we have an image, use the TensorFlow model for prediction
        if image_path:
            try:
                # Use the model_prediction function from app.py
                from app import model_prediction
                
                # Convert image_path to file-like object for the model
                with open(image_path, 'rb') as image_file:
                    result_index = model_prediction(image_file)
                
                if result_index is not None:
                    # Class names matching the user's model
                    class_name = [
                        'Apple - Apple Scab',
                        'Apple - Black Rot',
                        'Apple - Cedar Apple Rust',
                        'Apple - Healthy',
                        'Blueberry - Healthy',
                        'Cherry - Powdery Mildew',
                        'Cherry - Healthy',
                        'Corn - Cercospora Leaf Spot',
                        'Corn - Common Rust',
                        'Corn - Northern Leaf Blight',
                        'Corn - Healthy',
                        'Grape - Black Rot',
                        'Grape - Esca (Black Measles)',
                        'Grape - Leaf Blight',
                        'Grape - Healthy',
                        'Orange - Huanglongbing (Citrus Greening)',
                        'Peach - Bacterial Spot',
                        'Peach - Healthy',
                        'Bell Pepper - Bacterial Spot',
                        'Bell Pepper - Healthy',
                        'Potato - Early Blight',
                        'Potato - Late Blight',
                        'Potato - Healthy',
                        'Raspberry - Healthy',
                        'Soybean - Healthy',
                        'Squash - Powdery Mildew',
                        'Strawberry - Leaf Scorch',
                        'Strawberry - Healthy',
                        'Tomato - Bacterial Spot',
                        'Tomato - Early Blight',
                        'Tomato - Late Blight',
                        'Tomato - Leaf Mold',
                        'Tomato - Septoria Leaf Spot',
                        'Tomato - Spider Mites',
                        'Tomato - Target Spot',
                        'Tomato - Yellow Leaf Curl Virus',
                        'Tomato - Mosaic Virus',
                        'Tomato - Healthy'
                    ]
                    
                    diagnosis = class_name[result_index]
                    plant, condition = diagnosis.split(" - ")
                    
                    # Determine if healthy or diseased
                    is_healthy = "Healthy" in condition
                    confidence = 95.0 if is_healthy else 85.0  # High confidence for model predictions
                    
                    # Get disease details from our database if available
                    disease_key = None
                    for key, data in self.disease_data.items():
                        if condition.lower() in key.lower() or key.lower() in condition.lower():
                            disease_key = key
                            break
                    
                    details = self.disease_data.get(disease_key, {})
                    
                    diseases.append({
                        'disease': condition.replace('_', ' ').title(),
                        'confidence': confidence,
                        'details': details,
                        'method': 'AI Model Prediction',
                        'reasoning': [f'Model detected {condition} in {plant}'],
                        'matched_symptoms': [],
                        'type': details.get('type', 'Unknown')
                    })
                    
                    # If model detects disease, return immediately
                    if not is_healthy:
                        return diseases
                    else:
                        # For healthy plants, also check symptoms if provided
                        pass
                        
            except Exception as e:
                print(f"Model prediction failed: {e}")
                # Fall back to rule-based if model fails

        # Fallback to rule-based detection if no image or model failed
        symptom_diseases = self.rule_based_detection(crop_type, symptoms or [], severity)
        diseases.extend(symptom_diseases)
        
        return diseases

    def get_crop_specific_symptoms(self, crop_type):
        crop_specific_symptoms = {
            'leaf_symptoms': [],
            'plant_symptoms': [],
            'environmental_indicators': []
        }

        # Normalize crop type for matching
        crop_type_lower = crop_type.lower().strip()

        for disease, data in self.disease_data.items():
            # Check if this disease affects the selected crop
            affected_crops_lower = [c.lower().strip() for c in data['crop_affected']]

            # More flexible crop matching
            crop_matched = False
            for affected_crop in affected_crops_lower:
                if (affected_crop in crop_type_lower or
                    crop_type_lower in affected_crop or
                    affected_crop.replace(' ', '') in crop_type_lower.replace(' ', '') or
                    crop_type_lower.replace(' ', '') in affected_crop.replace(' ', '')):
                    crop_matched = True
                    break

            if crop_matched:
                # Add disease-specific symptoms to appropriate categories
                for symptom in data['symptoms']:
                    symptom_lower = symptom.lower()

                    # Categorize symptoms
                    if any(word in symptom_lower for word in ['leaf', 'leaves', 'spot', 'lesion', 'curl', 'yellow', 'brown', 'ring', 'halo', 'powdery', 'coating', 'mold']):
                        if symptom not in crop_specific_symptoms['leaf_symptoms']:
                            crop_specific_symptoms['leaf_symptoms'].append(symptom)
                    elif any(word in symptom_lower for word in ['plant', 'growth', 'stunted', 'wilting', 'death', 'drop', 'deformed', 'size']):
                        if symptom not in crop_specific_symptoms['plant_symptoms']:
                            crop_specific_symptoms['plant_symptoms'].append(symptom)
                    else:
                        # Environmental or general symptoms
                        if symptom not in crop_specific_symptoms['environmental_indicators']:
                            crop_specific_symptoms['environmental_indicators'].append(symptom)

        # Also add general symptoms that might be relevant
        general_leaf_symptoms = [
            'Yellowing of leaves', 'Brown spots on leaves', 'Leaf curling',
            'White powdery coating', 'Dark lesions', 'Premature leaf drop'
        ]

        general_plant_symptoms = [
            'Stunted growth', 'Wilting despite watering', 'Plant death',
            'Reduced yield', 'Flower drop', 'Fruit deformities'
        ]

        general_env_symptoms = [
            'Symptoms worse after rainfall', 'Symptoms appear in humid conditions',
            'Rapid symptom spread', 'Symptoms on lower leaves first'
        ]

        # Add general symptoms if they're not already covered by crop-specific ones
        for symptom in general_leaf_symptoms:
            if symptom not in crop_specific_symptoms['leaf_symptoms']:
                crop_specific_symptoms['leaf_symptoms'].append(symptom)

        for symptom in general_plant_symptoms:
            if symptom not in crop_specific_symptoms['plant_symptoms']:
                crop_specific_symptoms['plant_symptoms'].append(symptom)

        for symptom in general_env_symptoms:
            if symptom not in crop_specific_symptoms['environmental_indicators']:
                crop_specific_symptoms['environmental_indicators'].append(symptom)

        return crop_specific_symptoms

    def get_causal_analysis(self, disease, crop_type=None, severity='medium'):
        """Enhanced causal AI explanation for disease with visualizations"""
        if disease not in self.disease_data:
            return """
            Disease Analysis Not Available

            The specified disease is not in our database. Please consult with a local agricultural expert or extension service for proper diagnosis and treatment recommendations.

            Key points to consider:
            - Take clear photos of affected plant parts
            - Note environmental conditions (temperature, humidity, rainfall)
            - Consider recent weather patterns and irrigation practices
            - Check for insect activity or other stress factors
            """

        data = self.disease_data[disease]

        # Get pesticide recommendations if crop_type is provided
        pesticide_info = ""
        if crop_type and 'treatment' in data:
            treatment_data = data['treatment']

            # Determine severity level
            if severity == 'low':
                severity_key = 'low_severity'
            elif severity == 'medium':
                severity_key = 'medium_severity'
            else:  # high severity
                severity_key = 'high_severity'

            if severity_key in treatment_data and crop_type in treatment_data[severity_key]['pesticides']:
                pesticides = treatment_data[severity_key]['pesticides'][crop_type]
                frequency = treatment_data[severity_key]['frequency']

                pesticide_info = f"""
        💊 CROP-SPECIFIC PESTICIDE RECOMMENDATIONS:
        📅 Application Schedule: {frequency}

        Recommended Pesticides for {crop_type.title()}:
        {chr(10).join(f'• {pesticide}' for pesticide in pesticides)}

        ⚠️ SAFETY NOTES:
        • Always follow label instructions and local regulations
        • Apply during early morning or evening hours
        • Wear protective equipment (gloves, mask, protective clothing)
        • Avoid application during windy conditions
        • Keep children and pets away from treated areas
        """

        # Enhanced causal analysis with more detail
        analysis = f"""
        🌾 CAUSAL AI ANALYSIS: {disease.replace('_', ' ').title()}

        🔍 ROOT CAUSE ANALYSIS:
        {chr(10).join(f'• {cause}' for cause in data['causes'])}

        ⚠️ POTENTIAL EFFECTS IF UNTReATED:
        {chr(10).join(f'• {effect}' for effect in data['effects'])}

        🛡️ PREVENTION STRATEGIES:
        {chr(10).join(f'• {prevention}' for prevention in data['prevention'])}

        💊 GENERAL TREATMENT APPROACH:
        {pesticide_info if pesticide_info else "Consult agricultural experts for specific treatment recommendations based on your crop type and severity level."}

        📊 RISK ASSESSMENT DASHBOARD:
        ┌─────────────────────────────────────┐
        │ Disease Type:     {data['type']:<15} │
        │ Spread Rate:      {data['spread_rate']:<15} │
        │ Economic Impact:  {data['economic_impact']:<15} │
        │ Treatment Cost:   {data['treatment_cost']:<15} │
        └─────────────────────────────────────┘

        🎯 ACTION PLAN:
        1. Immediately isolate affected plants
        2. Apply recommended treatments within 24-48 hours
        3. Monitor nearby plants for early symptoms
        4. Implement prevention measures for next season
        5. Document conditions for future reference

        📞 PROFESSIONAL CONSULTATION:
        For severe outbreaks or uncertain diagnosis, consult:
        • Local agricultural extension service
        • Certified plant pathologist
        • Agricultural university specialists

        📈 MONITORING CHECKLIST:
        □ Daily inspection of affected area
        □ Track symptom progression
        □ Monitor weather conditions
        □ Document treatment effectiveness
        □ Plan preventive measures for next season
        """

        return analysis.strip()

    def get_pesticide_recommendations(self, disease, crop_type, severity):
        """Get crop-specific and severity-based pesticide recommendations"""
        if disease not in self.disease_data or 'treatment' not in self.disease_data[disease]:
            return None

        treatment_data = self.disease_data[disease]['treatment']

        # Determine severity level
        if severity == 'low':
            severity_key = 'low_severity'
        elif severity == 'medium':
            severity_key = 'medium_severity'
        else:  # high severity
            severity_key = 'high_severity'

        if severity_key in treatment_data and crop_type in treatment_data[severity_key]['pesticides']:
            return {
                'frequency': treatment_data[severity_key]['frequency'],
                'pesticides': treatment_data[severity_key]['pesticides'][crop_type]
            }

        return None

    def simple_detect_disease(self, crop, symptoms):
        """Simple rule-based disease detection"""
        detected = []
        for rule in self.disease_rules:
            if rule["crop"] == crop:
                match_count = len(set(symptoms) & set(rule["symptoms"]))
                if match_count > 0:
                    detected.append({
                        "disease": rule["disease"],
                        "pesticide": rule["pesticide"],
                        "cost_per_ha": rule["cost_per_ha"],
                        "matched_symptoms": list(set(symptoms) & set(rule["symptoms"]))
                    })
        return detected

    def simple_causal_analysis(self, detected):
        """Simple causal analysis"""
        causal_info = []
        for r in detected:
            info = f"Disease {r['disease']} triggered because symptoms {r['matched_symptoms']} matched known patterns."
            causal_info.append(info)
        return causal_info

# Usage example
if __name__ == "__main__":
    # Test crop recommendation
    recommender = CropRecommender()
    soil_data = {'N': 75, 'P': 35, 'K': 35, 'ph': 6.8, 'temperature': 24, 'humidity': 75, 'rainfall': 120}
    recommendations = recommender.recommend_crops(soil_data, "India")

    print("Crop Recommendations:")
    for rec in recommendations:
        print(f"- {rec['crop'].title()}: {rec['confidence']}% confidence ({rec.get('method', 'Rule-based')})")

    # Test disease detection
    detector = DiseaseDetector()
    results = detector.simple_detect_disease("Tomato", ["yellow leaves", "spots on leaves"])

    print("\nDisease Detection Results:")
    for r in results:
        print(f"Crop: Tomato")
        print(f"Disease Detected: {r['disease']}")
        print(f"Pesticide Recommended: {r['pesticide']}")
        print(f"Cost per hectare: ${r['cost_per_ha']}")
        print(f"Matched Symptoms: {r['matched_symptoms']}")

    causal_explanations = detector.simple_causal_analysis(results)
    print("\nCausal Explanations:")
    for e in causal_explanations:
        print("-", e)
