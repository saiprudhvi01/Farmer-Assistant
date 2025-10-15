import os
import difflib
import json
import numpy as np
import pandas as pd
import pickle
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from src.forms import RegistrationForm, LoginForm
# Import models after set_db() is called
from src.models.user import set_db
from src.models.crop_recommendation import CropRecommender
import re
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import google.generativeai as genai

# Import configuration
from config import config

# Initialize Flask app
app = Flask(__name__)

# Load configuration based on environment variable or default to development
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Import extensions after Flask app creation
from src.extensions import db

# Initialize extensions with the Flask app
db.init_app(app)

# Initialize User model with database instance
from src.models.user import set_db
set_db(db)

# Now import the models after set_db() has been called
from src.models.user import User, CropListing, Offer

# Create database tables
with app.app_context():
    db.create_all()

# Add custom Jinja2 filter for number formatting
def format_number(value):
    """Format number with comma as thousand separator."""
    if value is None:
        return ""
    try:
        return "{:,}".format(int(value))
    except (ValueError, TypeError):
        return value

app.jinja_env.filters['number_format'] = format_number

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
crop_recommender = CropRecommender()

# Sample crop data (replace with your actual data)
CROP_DATA = {
    'rice': {
        'name': 'Rice',
        'season': 'Kharif',
        'soil_type': 'Clay loam',
        'min_temp': 20,
        'max_temp': 35,
        'rainfall': 1500,
        'ph_range': (5.0, 7.5)
    },
    'wheat': {
        'name': 'Wheat',
        'season': 'Rabi',
        'soil_type': 'Loam',
        'min_temp': 10,
        'max_temp': 25,
        'rainfall': 750,
        'ph_range': (6.0, 7.5)
    },
    # Add more crops as needed
}

# Historical price data for trends (last 30 days)
PRICE_HISTORY = {
    'rice': [
        {'date': '2024-01-01', 'price': 40.0},
        {'date': '2024-01-02', 'price': 41.0},
        {'date': '2024-01-03', 'price': 42.0},
        {'date': '2024-01-04', 'price': 43.0},
        {'date': '2024-01-05', 'price': 44.0},
        {'date': '2024-01-06', 'price': 45.0}
    ],
    'wheat': [
        {'date': '2024-01-01', 'price': 38.0},
        {'date': '2024-01-02', 'price': 37.0},
        {'date': '2024-01-03', 'price': 36.0},
        {'date': '2024-01-04', 'price': 35.0},
        {'date': '2024-01-05', 'price': 35.0},
        {'date': '2024-01-06', 'price': 35.0}
    ],
    'cotton': [
        {'date': '2024-01-01', 'price': 50.0},
        {'date': '2024-01-02', 'price': 51.0},
        {'date': '2024-01-03', 'price': 52.0},
        {'date': '2024-01-04', 'price': 53.0},
        {'date': '2024-01-05', 'price': 54.0},
        {'date': '2024-01-06', 'price': 55.0}
    ]
}

# Sample forum posts
FORUM_POSTS = [
    {
        'id': 1,
        'title': 'Best practices for organic farming',
        'content': 'Sharing some tips on organic farming methods that have worked well for me...',
        'author': 'Organic Farmer',
        'date': '2025-10-07',
        'comments': [
            {'author': 'John', 'content': 'Thanks for sharing!', 'date': '2025-10-07'},
            {'author': 'Ravi', 'content': 'Any specific tips for tomato farming?', 'date': '2025-10-08'}
        ],
        'likes': 15,
        'user_type': 'farmer'
    },
    # Add more sample posts
]

# Market data for trends with local and global prices
MARKET_DATA = [
    {'crop': 'Rice', 'variety': 'Basmati', 'price': 45.0, 'unit': 'kg', 'location': 'Karnal', 'date': '2025-10-08', 'farmer_id': 1, 'is_available': True},
    {'crop': 'Wheat', 'variety': 'Sharbati', 'price': 35.0, 'unit': 'kg', 'location': 'Punjab', 'date': '2025-10-08', 'farmer_id': 2, 'is_available': True},
    {'crop': 'Tomato', 'variety': 'Hybrid', 'price': 60.0, 'unit': 'kg', 'location': 'Nashik', 'date': '2025-10-08', 'farmer_id': 3, 'is_available': True},
    {'crop': 'Cotton', 'variety': 'BT Cotton', 'price': 55.0, 'unit': 'kg', 'location': 'Gujarat', 'date': '2025-10-08', 'farmer_id': 4, 'is_available': True},
    {'crop': 'Maize', 'variety': 'Hybrid', 'price': 30.0, 'unit': 'kg', 'location': 'Karnataka', 'date': '2025-10-08', 'farmer_id': 5, 'is_available': True},
]

# NASA Power API integration for weather data
def get_location_coordinates(location, api_key="fd4792498aba439d841c4d0ed3717d7c"):
    """Get latitude and longitude for a given location using OpenCage Geocoding API"""
    try:
        url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data['results']:
                geometry = data['results'][0]['geometry']
                return geometry['lat'], geometry['lng']
            else:
                return None, None
        else:
            return None, None
    except requests.exceptions.RequestException:
        return None, None

def get_nasa_weather_data(lat, lon, date=None):
    """Get weather data from NASA Power API"""
    try:
        # Validate coordinates
        if not (-90 <= lat <= 90):
            return None
        if not (-180 <= lon <= 180):
            return None

        # NASA Power API endpoint
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"

        if date is None:
            # Use a date from 2 months ago to ensure data availability
            target_date = datetime.now() - timedelta(days=70)
            date = target_date.strftime("%Y%m%d")

        # Essential parameters for crop recommendation
        params = {
            'parameters': 'T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M,PS,ALLSKY_SFC_SW_DWN',
            'community': 'AG',
            'longitude': float(lon),
            'latitude': float(lat),
            'start': date,
            'end': date,
            'format': 'JSON'
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'error' in data:
                return None
            return data
        else:
            return None

    except Exception:
        return None

def process_nasa_weather_data(nasa_data):
    """Process NASA weather data into format suitable for crop recommendation"""
    try:
        properties = nasa_data['properties']
        parameter_data = properties['parameter']

        # Helper function to extract parameter value safely
        def get_param_value(param_name, default_value):
            param_dict = parameter_data.get(param_name, {})
            if param_dict and isinstance(param_dict, dict):
                values = list(param_dict.values())
                if values:
                    value = values[0]
                    if value == -999.0 or value == -999:  # NASA Power missing data flag
                        return default_value
                    return float(value)
            return default_value

        # Extract weather parameters
        temperature = get_param_value('T2M', 25)
        humidity = get_param_value('RH2M', 60)
        rainfall = get_param_value('PRECTOTCORR', 0) * 365  # Convert to annual
        wind_speed = get_param_value('WS2M', 5)
        surface_pressure = get_param_value('PS', 101.3) * 10  # Convert to hPa

        # Validate values
        if not (0 <= humidity <= 100):
            humidity = 60
        if not (0 <= rainfall <= 5000):
            rainfall = 1000
        if not (0 <= wind_speed <= 50):
            wind_speed = 5
        if not (900 <= surface_pressure <= 1100):
            surface_pressure = 1013

        return {
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': wind_speed,
            'pressure': surface_pressure,
            'data_source': 'NASA Power API'
        }

    except Exception:
        return None

def get_crop_recommendation_from_api(location):
    """Get crop recommendation using NASA Power API weather data"""
    try:
        # Get coordinates
        lat, lon = get_location_coordinates(location)
        if lat is None or lon is None:
            return None

        # Get NASA weather data
        nasa_data = get_nasa_weather_data(lat, lon)
        if nasa_data is None:
            return None

        # Process weather data
        weather_data = process_nasa_weather_data(nasa_data)
        if weather_data is None:
            return None

        # Simple crop recommendation logic based on weather data
        recommendations = []

        # Rice recommendation (more flexible conditions for tropical regions)
        if (weather_data['temperature'] >= 15 and weather_data['temperature'] <= 40 and
            weather_data['rainfall'] >= 500 and weather_data['humidity'] >= 40):
            recommendations.append({
                'crop': 'Rice',
                'confidence': min(95, 60 + (weather_data['rainfall'] / 1000 * 20) + (weather_data['humidity'] / 50 * 15)),
                'reason': 'High rainfall and suitable temperature range for rice cultivation',
                'weather_match': {
                    'temperature': f"{weather_data['temperature']:.1f}°C (optimal: 20-35°C)",
                    'rainfall': f"{weather_data['rainfall']:.0f}mm (optimal: >1000mm)",
                    'humidity': f"{weather_data['humidity']:.1f}% (optimal: >50%)"
                }
            })

        # Wheat recommendation (for cooler regions)
        if (weather_data['temperature'] >= 5 and weather_data['temperature'] <= 30 and
            weather_data['rainfall'] >= 300 and weather_data['rainfall'] <= 1500):
            recommendations.append({
                'crop': 'Wheat',
                'confidence': min(90, 50 + (25 - abs(weather_data['temperature'] - 15)) * 2 + min(weather_data['rainfall'] / 1000 * 20, 20)),
                'reason': 'Moderate rainfall and cooler temperature suitable for wheat',
                'weather_match': {
                    'temperature': f"{weather_data['temperature']:.1f}°C (optimal: 10-25°C)",
                    'rainfall': f"{weather_data['rainfall']:.0f}mm (optimal: 500-1000mm)",
                    'humidity': f"{weather_data['humidity']:.1f}% (moderate humidity preferred)"
                }
            })

        # Maize recommendation (versatile crop)
        if (weather_data['temperature'] >= 12 and weather_data['temperature'] <= 35 and
            weather_data['rainfall'] >= 400 and weather_data['rainfall'] <= 1500):
            recommendations.append({
                'crop': 'Maize',
                'confidence': min(88, 55 + min(weather_data['rainfall'] / 1200 * 20, 20) + (weather_data['humidity'] / 60 * 13)),
                'reason': 'Warm temperature and adequate rainfall for maize growth',
                'weather_match': {
                    'temperature': f"{weather_data['temperature']:.1f}°C (optimal: 18-32°C)",
                    'rainfall': f"{weather_data['rainfall']:.0f}mm (optimal: 600-1200mm)",
                    'humidity': f"{weather_data['humidity']:.1f}% (moderate humidity suitable)"
                }
            })

        # Tomato recommendation (for moderate climates)
        if (weather_data['temperature'] >= 10 and weather_data['temperature'] <= 35 and
            weather_data['rainfall'] >= 300 and weather_data['rainfall'] <= 1000):
            recommendations.append({
                'crop': 'Tomato',
                'confidence': min(85, 50 + (30 - abs(weather_data['temperature'] - 22)) * 1.5 + min(weather_data['rainfall'] / 800 * 20, 20)),
                'reason': 'Moderate temperature and rainfall suitable for tomato cultivation',
                'weather_match': {
                    'temperature': f"{weather_data['temperature']:.1f}°C (optimal: 15-30°C)",
                    'rainfall': f"{weather_data['rainfall']:.0f}mm (optimal: 400-800mm)",
                    'humidity': f"{weather_data['humidity']:.1f}% (moderate humidity preferred)"
                }
            })

        # Cotton recommendation (for warm, dry conditions)
        if (weather_data['temperature'] >= 15 and weather_data['temperature'] <= 40 and
            weather_data['rainfall'] >= 400 and weather_data['rainfall'] <= 1200 and
            weather_data['humidity'] >= 30 and weather_data['humidity'] <= 80):
            recommendations.append({
                'crop': 'Cotton',
                'confidence': min(82, 45 + (35 - abs(weather_data['temperature'] - 27)) * 1.2 + min(weather_data['rainfall'] / 1000 * 15, 15) + (weather_data['humidity'] / 60 * 10)),
                'reason': 'Warm, dry conditions with moderate rainfall suitable for cotton',
                'weather_match': {
                    'temperature': f"{weather_data['temperature']:.1f}°C (optimal: 20-35°C)",
                    'rainfall': f"{weather_data['rainfall']:.0f}mm (optimal: 500-1000mm)",
                    'humidity': f"{weather_data['humidity']:.1f}% (lower humidity preferred)"
                }
            })

        # Ensure we have at least some recommendations
        if not recommendations:
            # Fallback: recommend the most suitable crops based on temperature alone
            if weather_data['temperature'] >= 20 and weather_data['temperature'] <= 35:
                recommendations.append({
                    'crop': 'Rice',
                    'confidence': 65,
                    'reason': 'Temperature conditions are suitable for rice cultivation',
                    'weather_match': {
                        'temperature': f"{weather_data['temperature']:.1f}°C (suitable range)",
                        'rainfall': f"{weather_data['rainfall']:.0f}mm (may need irrigation)",
                        'humidity': f"{weather_data['humidity']:.1f}% (adequate)"
                    }
                })
            elif weather_data['temperature'] >= 15 and weather_data['temperature'] <= 25:
                recommendations.append({
                    'crop': 'Wheat',
                    'confidence': 60,
                    'reason': 'Temperature conditions are suitable for wheat cultivation',
                    'weather_match': {
                        'temperature': f"{weather_data['temperature']:.1f}°C (suitable range)",
                        'rainfall': f"{weather_data['rainfall']:.0f}mm (may need irrigation)",
                        'humidity': f"{weather_data['humidity']:.1f}% (adequate)"
                    }
                })
            else:
                # Default recommendation for any location
                recommendations.append({
                    'crop': 'Tomato',
                    'confidence': 55,
                    'reason': 'Versatile crop that can grow in various conditions',
                    'weather_match': {
                        'temperature': f"{weather_data['temperature']:.1f}°C (moderate range)",
                        'rainfall': f"{weather_data['rainfall']:.0f}mm (can be irrigated)",
                        'humidity': f"{weather_data['humidity']:.1f}% (moderate humidity)"
                    }
                })

        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'location': location,
            'coordinates': {'lat': lat, 'lon': lon},
            'weather_data': weather_data,
            'recommendations': recommendations[:3],  # Top 3 recommendations
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        return {'error': str(e)}

# Authentication routes
@app.route('/')
def index():
    """Home page with overview of features."""
    return render_template('index.html',
                         crops=CROP_DATA.values(),
                         market_data=MARKET_DATA[:5],
                         posts=FORUM_POSTS[:3])

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    form = RegistrationForm()

    if form.validate_on_submit():
        # Check if terms are accepted
        if not form.accept_terms.data:
            flash('You must accept the Terms of Service and Privacy Policy to register.', 'danger')
            return render_template('register.html', form=form)

        # Create user based on type
        if form.user_type.data == 'farmer':
            user = User(
                username=form.username.data,
                email=form.email.data,
                full_name=f"{form.first_name.data} {form.last_name.data}",
                phone=form.phone.data,
                city=form.location.data,
                user_type='farmer',
                farm_size=form.farm_size.data,
                primary_crops=f"Experience: {form.experience.data} years, Specialization: {form.specialization.data}" if form.experience.data else form.specialization.data
            )
        else:
            user = User(
                username=form.username.data,
                email=form.email.data,
                full_name=f"{form.first_name.data} {form.last_name.data}",
                phone=form.phone.data,
                city=form.location.data,
                user_type='buyer',
                business_name=form.business_name.data,
                business_type=form.business_type.data,
                buying_capacity=form.buying_capacity.data
            )

        try:
            db.session.add(user)
            db.session.commit()

            flash('Registration successful! Please login to continue.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            print(f"DEBUG: Registration failed with error: {str(e)}")
            print(f"DEBUG: Error type: {type(e)}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            flash(f'Registration failed: {str(e)}', 'danger')

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    form = LoginForm()

    if form.validate_on_submit():
        # Find user by username or email
        user = User.query.filter(
            (User.username == form.username.data) | (User.email == form.username.data)
        ).first()

        if user:
            # Redirect based on user type
            if user.user_type == 'farmer':
                flash(f'Welcome, {user.get_full_name()}!', 'success')
                return redirect(url_for('farmer_dashboard'))
            else:
                flash(f'Welcome, {user.get_full_name()}!', 'success')
                return redirect(url_for('marketplace'))
        else:
            flash('User not found. Please register first.', 'danger')

    return render_template('login.html', form=form)

@app.route('/farmer-dashboard')
def farmer_dashboard():
    """Farmer dashboard showing their listings."""
    # Get the current logged-in farmer (in real app, this would be current_user)
    farmer = User.query.filter_by(user_type='farmer').first()

    # Initialize variables to avoid template errors
    listings = []

    if farmer:
        # Get farmer's crop listings
        listings = CropListing.query.filter_by(farmer_id=farmer.id, is_available=True).all()

    return render_template('farmer_dashboard.html',
                         farmer=farmer,
                         listings=listings)

@app.route('/api/add-listing', methods=['POST'])
def add_listing():
    """API endpoint for farmers to add crop listings."""
    # Get a sample farmer for demo
    farmer = User.query.filter_by(user_type='farmer').first()
    if not farmer:
        return jsonify({'error': 'No farmers found. Please register as a farmer first.'}), 400

    data = request.get_json()

    try:
        listing = CropListing(
            farmer_id=farmer.id,
            crop_name=data['crop_name'],
            variety=data.get('variety', ''),
            quantity=float(data['quantity']),
            unit=data['unit'],
            price_per_unit=float(data['price_per_unit']),
            location=data['location'],
            description=data.get('description', ''),
            is_organic=data.get('is_organic', False)
        )

        db.session.add(listing)
        db.session.commit()

        return jsonify({
            'success': True,
            'listing_id': listing.id,
            'message': 'Listing added successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/listing/<int:listing_id>')
def get_listing(listing_id):
    """API endpoint to get listing details."""
    listing = CropListing.query.get_or_404(listing_id)

    return jsonify({
        'id': listing.id,
        'crop_name': listing.crop_name,
        'variety': listing.variety,
        'quantity': listing.quantity,
        'unit': listing.unit,
        'price_per_unit': listing.price_per_unit,
        'location': listing.location,
        'description': listing.description,
        'is_organic': listing.is_organic,
        'farmer_name': listing.farmer.get_full_name(),
        'farmer_phone': listing.farmer.phone,
        'farmer_location': listing.farmer.location
    })

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    """Crop recommendation based on location and soil conditions."""
    if request.method == 'POST':
        # Get form data
        location = request.form.get('location')
        soil_type = request.form.get('soil_type')
        ph = float(request.form.get('ph', 7.0))
        rainfall = float(request.form.get('rainfall', 1000))
        temperature = float(request.form.get('temperature', 25))

        # Use the new crop recommendation model
        soil_data = {
            'ph': ph,
            'rainfall': rainfall,
            'temperature': temperature,
            'soil_type': soil_type
        }

        recommended_crops = crop_recommender.recommend_crops(soil_data, location)

        # Generate chart data for visualizations
        chart_data = {
            'labels': [crop['crop'] for crop in recommended_crops],
            'datasets': [{
                'label': 'Confidence (%)',
                'data': [crop['confidence'] for crop in recommended_crops],
                'backgroundColor': ['rgba(75, 192, 192, 0.2)', 'rgba(255, 99, 132, 0.2)', 'rgba(255, 205, 86, 0.2)'],
                'borderColor': ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)', 'rgba(255, 205, 86, 1)'],
                'borderWidth': 1
            }]
        }

        return render_template('crop_recommendation.html',
                             recommended_crops=recommended_crops,
                             form_data=request.form,
                             chart_data=chart_data)

    return render_template('crop_recommendation.html')

@app.route('/crop-recommendation-api', methods=['GET', 'POST'])
def crop_recommendation_api():
    """Crop recommendation using NASA Power API weather data."""
    if request.method == 'POST':
        location = request.form.get('location', '').strip()

        if not location:
            flash('Please enter a location', 'warning')
            return redirect(request.url)

        # Get crop recommendation using NASA Power API
        api_result = get_crop_recommendation_from_api(location)

        if api_result and 'error' not in api_result:
            if api_result['recommendations']:
                return render_template('crop_recommendation_api.html',
                                     api_result=api_result,
                                     location=location)
            else:
                flash(f'No suitable crops found for {location}. Weather conditions may not be suitable for common crops.', 'info')
                return render_template('crop_recommendation_api.html',
                                     api_result=None,
                                     location=location)
        else:
            error_msg = api_result.get('error', 'Unable to get weather data for this location') if api_result else 'Unable to process location'
            flash(f'Error: {error_msg}. Please try a different location or check your internet connection.', 'error')
            return render_template('crop_recommendation_api.html',
                                 api_result=None,
                                 location=location)

    return render_template('crop_recommendation_api.html', api_result=None, location='')

@app.route('/api/crop-recommendation/<location>')
def api_crop_recommendation(location):
    """API endpoint for crop recommendation using NASA Power API."""
    try:
        result = get_crop_recommendation_from_api(location)

        if result and 'error' not in result:
            return jsonify(result)
        else:
            return jsonify({
                'error': 'Unable to get crop recommendation',
                'location': location
            }), 400

    except Exception as e:
        return jsonify({
            'error': str(e),
            'location': location
        }), 500

@app.route('/marketplace')
def marketplace():
    """Display the marketplace for farmers to manage their listings."""
    # Get sample farmer for demo
    farmer = User.query.filter_by(user_type='farmer').first()

    if farmer:
        # Farmers see their own listings and can add new ones
        listings = CropListing.query.filter_by(farmer_id=farmer.id).all()
        return render_template('marketplace_farmer.html',
                             listings=listings,
                             title="My Marketplace",
                             farmer=farmer)
    else:
        # No farmers yet - show empty state
        return render_template('marketplace_farmer.html',
                             listings=[],
                             title="Marketplace",
                             farmer=None)

@app.route('/market-trends')
@app.route('/market-trends/<location>')
def market_trends(location=None):
    """Display market trends and price analysis for crops based on location."""
    from datetime import datetime

    # Handle both query parameter and path parameter
    if location is None:
        location = request.args.get('location')

    # If no location provided, show general market trends
    if location is None:
        return render_template('market_trends.html',
                            market_data=MARKET_DATA,
                            price_history=PRICE_HISTORY,
                            title="Market Trends",
                            location=None,
                            now=datetime.utcnow())

    # Get crop recommendations for the location
    print(f"DEBUG: Getting crop recommendations for location: {location}")
    api_result = get_crop_recommendation_from_api(location)
    print(f"DEBUG: API result: {api_result}")

    if api_result and 'error' not in api_result:
        print(f"DEBUG: API result has {len(api_result.get('recommendations', []))} recommendations")
        if api_result.get('recommendations'):
            # Get recommended crops for this location
            recommended_crops = [rec['crop'] for rec in api_result['recommendations']]
            print(f"DEBUG: Recommended crops: {recommended_crops}")

            # Filter market data to show only crops suitable for this location
            location_market_data = []
            for item in MARKET_DATA:
                if item['crop'] in recommended_crops:
                    # Create a location-specific version of the market item
                    location_item = item.copy()
                    location_item['location'] = location  # Update location to user's location
                    location_item['suitability'] = 'High'  # Mark as suitable for this location
                    location_market_data.append(location_item)

            print(f"DEBUG: Found {len(location_market_data)} matching market items")

            # If no market data for recommended crops, show the recommended crops with estimated prices
            if not location_market_data:
                print("DEBUG: No market data found, creating estimated items")
                location_market_data = []
                for rec in api_result['recommendations']:
                    # Create estimated market data based on crop recommendations
                    estimated_item = {
                        'crop': rec['crop'],
                        'variety': 'Local Variety',
                        'price': 40.0 + (rec['confidence'] * 0.5),  # Price based on confidence
                        'unit': 'kg',
                        'location': location,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'farmer_id': None,
                        'is_available': True,
                        'suitability': 'High',
                        'confidence': rec['confidence'],
                        'reason': rec['reason']
                    }
                    location_market_data.append(estimated_item)

            return render_template('market_trends.html',
                                market_data=location_market_data,
                                price_history=PRICE_HISTORY,
                                title=f"Market Trends - {location}",
                                location=location,
                                api_result=api_result,
                                now=datetime.utcnow())

    # If location-based data not available, fall back to general market data
    print(f"DEBUG: Location-based data not available, showing general market trends")
    flash(f'Could not get location-specific data for {location}. Showing general market trends.', 'info')
    return render_template('market_trends.html',
                        market_data=MARKET_DATA,
                        price_history=PRICE_HISTORY,
                        title="Market Trends",
                        location=location,
                        now=datetime.utcnow())

@app.route('/community')
def community():
    """Community forum for farmers."""
    return render_template('community.html',
                         posts=FORUM_POSTS,
                         title="Community Forum")

@app.route('/api/weather/<location>')
def get_weather(location):
    """API endpoint to get weather data for a location."""
    # In a real app, this would fetch from a weather API
    return jsonify({
        'location': location,
        'temperature': 28.5,
        'humidity': 65,
        'rainfall': 15,
        'forecast': 'Partly cloudy with a chance of rain'
    })

@app.route('/api/market-prices')
def get_market_prices():
    """API endpoint to get market prices."""
    return jsonify(MARKET_DATA)

@app.route('/api/like-post/<int:post_id>', methods=['POST'])
def like_post(post_id):
    """Handle post likes."""
    # In a real app, this would update the database
    return jsonify({'success': True, 'likes': FORUM_POSTS[post_id-1]['likes'] + 1})

from datetime import datetime
import json
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Define DiseaseDetector class here for separation
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
        """Generate causal analysis for detected diseases"""
        analysis = []
        for r in results:
            pesticide = r.get('pesticide', 'Consult expert for appropriate pesticide')
            cost = r.get('cost_per_ha', 0)
            
            para = f"""
Causal Analysis for {crop}:
The disease '{r['disease']}' was detected primarily due to the presence of symptoms {r['matched_symptoms']}. 
These symptoms indicate stress or infection in the crop that, if left unchecked, could lead to significant yield loss, stunted growth, or complete crop failure. 
Immediate application of the recommended pesticide '{pesticide}' at a cost of ${cost} per hectare is advised to mitigate the spread. 
Monitoring and preventive measures should follow to avoid recurrence, ensuring crop health and productivity.
"""
            analysis.append(para.strip())
        return analysis

# Initialize disease detector
disease_detector = DiseaseDetector()

@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    """Disease detection based on crop symptoms."""
    # Create symptoms dict for template
    symptoms_dict = {}
    for rule in disease_detector.disease_rules:
        crop = rule['crop'].lower()
        if crop not in symptoms_dict:
            symptoms_dict[crop] = []
        symptoms_dict[crop].extend(rule['symptoms'])
    # Remove duplicates
    for crop in symptoms_dict:
        symptoms_dict[crop] = list(set(symptoms_dict[crop]))

    if request.method == 'POST':
        crop = request.form.get('crop')
        symptoms = [s.strip().lower() for s in request.form.getlist('symptoms')]

        if not crop or not symptoms:
            flash('Please select a crop and at least one symptom', 'warning')
            return redirect(request.url)

        # Detect diseases
        results = disease_detector.detect_disease(crop, symptoms)

        if results:
            # Generate causal analysis
            analysis = disease_detector.causal_ai_analysis(crop, results)
            
            # Generate graph data for visualization
            graph_data = []
            for result in results:
                nodes = []
                edges = []
                
                # Add symptoms nodes
                for symptom in result['matched_symptoms']:
                    nodes.append({'id': symptom, 'label': symptom, 'group': 'symptom'})
                    edges.append({'from': symptom, 'to': result['disease']})
                
                # Add disease node
                nodes.append({'id': result['disease'], 'label': result['disease'], 'group': 'disease'})
                
                # Add pesticide node
                nodes.append({'id': result['pesticide'], 'label': result['pesticide'], 'group': 'pesticide'})
                edges.append({'from': result['disease'], 'to': result['pesticide']})
                
                # Add cost node
                cost_label = f"${result['cost_per_ha']}/ha"
                nodes.append({'id': cost_label, 'label': cost_label, 'group': 'cost'})
                edges.append({'from': result['pesticide'], 'to': cost_label})
                
                graph_data.append({
                    'nodes': nodes,
                    'edges': edges,
                    'title': f'Causal Graph for {result["disease"]}'
                })
            
            return render_template('disease_detection.html',
                                 results=results,
                                 crop=crop,
                                 symptoms=symptoms,
                                 analysis=analysis,
                                 symptoms_dict=symptoms_dict,
                                 graph_data=graph_data)
        else:
            flash('No diseases detected for the given symptoms. Please consult an expert.', 'info')
            return render_template('disease_detection.html', symptoms_dict=symptoms_dict)

    return render_template('disease_detection.html', symptoms_dict=symptoms_dict)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    """Rule-based chatbot for farming questions."""
    qa_pairs = [
        {"question": "How do I choose the best crop for my soil?", "answer": "Consider your soil type (clay, sandy, loamy), pH level, climate, and water availability. For example, rice thrives in clay soil with pH 6-7, while cotton prefers loamy soil with pH 7-8. Test your soil and use our Crop Recommendation page for personalized suggestions."},
        {"question": "What are common diseases in rice?", "answer": "Common rice diseases include Brown Spot (caused by fungus, symptoms: brown spots on leaves), Bacterial Leaf Blight (yellow stripes, wilting), and Blast (lesions on leaves). Prevent with resistant varieties, proper irrigation, and fungicides like Mancozeb."},
        {"question": "How to detect crop diseases?", "answer": "Look for symptoms like yellowing leaves, spots, wilting, stunted growth, or unusual patterns. Use our Disease Detection page to input symptoms and get diagnosis and treatment recommendations."},
        {"question": "Best pesticide for wheat rust?", "answer": "Propiconazole is effective for Leaf Rust in wheat. Apply at early symptoms, follow safety guidelines, and rotate with other fungicides to prevent resistance."},
        {"question": "Current market price for rice?", "answer": "Rice prices vary by region and quality (e.g., $0.50-$1 per kg in India). Check our Market Trends page for latest data or local markets for real-time prices."},
        {"question": "How to improve soil health?", "answer": "Add organic matter like compost, rotate crops, maintain pH 6-7, avoid over-fertilization, and use cover crops. Test soil regularly for nutrients."},
        {"question": "Symptoms of maize streak virus?", "answer": "Yellow streaks on leaves, stunted growth, cob rot. Transmitted by leafhoppers. Control vectors with insecticides like Imidacloprid and use resistant varieties."},
        {"question": "Suitable crops for rainy season?", "answer": "Rice, maize, sugarcane, and pulses like soybean are ideal for Kharif season due to high rainfall. Ensure good drainage to prevent waterlogging."},
        {"question": "How to use NASA API for crop advice?", "answer": "Input your location in our Crop Recommendation API page. It fetches weather data (temperature, rainfall) from NASA POWER API for accurate suggestions."},
        {"question": "Pesticide for tomato early blight?", "answer": "Chlorothalonil or Copper Oxychloride. Apply preventively, rotate chemicals, and remove infected plants to control spread."},
        # Extend to 100 with variations
    ]
    base_questions = [
        "How to prevent crop diseases?",
        "Best fertilizer for cotton?",
        "Market trends for sugarcane?",
        "Symptoms of potato late blight?",
        "Crop rotation tips?",
        "Soil pH for onion?",
        "Pesticide for soybean rust?",
        "Farming tips for groundnut?",
        "Disease detection methods?",
        "Crop recommendation based on rainfall?",
        "Common pests in wheat?",
        "How to increase crop yield?",
        "Organic farming practices?",
        "Weather impact on crops?",
        "Sustainable farming techniques?",
        "Irrigation methods for rice?",
        "Post-harvest handling?",
        "Market prices for vegetables?",
        "Disease resistant varieties?",
        "Soil testing importance?",
    ]
    for i, q in enumerate(base_questions, 11):
        answers = {
            "How to prevent crop diseases?": "Use resistant varieties, proper sanitation, crop rotation, balanced fertilization, and timely pesticide application.",
            "Best fertilizer for cotton?": "NPK 20:10:10 or urea for nitrogen, DAP for phosphorus. Apply based on soil test.",
            "Market trends for sugarcane?": "Sugarcane prices fluctuate; currently around $30-50 per ton. Check local trends.",
            "Symptoms of potato late blight?": "Dark spots on leaves, white mold underneath, rapid wilting. Control with Metalaxyl fungicide.",
            "Crop rotation tips?": "Rotate with legumes to fix nitrogen, avoid same family crops to prevent disease buildup.",
            "Soil pH for onion?": "Optimal pH 6.0-7.0. Lime if acidic, sulfur if alkaline.",
            "Pesticide for soybean rust?": "Trifloxystrobin or Tebuconazole. Apply at first sign of rust.",
            "Farming tips for groundnut?": "Plant in well-drained soil, irrigate regularly, harvest at maturity to avoid aflatoxin.",
            "Disease detection methods?": "Visual inspection, lab tests, symptom apps. Use our page for AI diagnosis.",
            "Crop recommendation based on rainfall?": "High rainfall: rice, maize. Low: millets, cotton. Use our tool for specifics.",
            "Common pests in wheat?": "Aphids, armyworms. Control with neem oil or insecticides.",
            "How to increase crop yield?": "Use quality seeds, optimal spacing, timely irrigation, balanced fertilizers, pest control.",
            "Organic farming practices?": "No chemicals, use compost, bio-pesticides, crop rotation, green manures.",
            "Weather impact on crops?": "Drought reduces yield, excess rain causes lodging. Monitor forecasts.",
            "Sustainable farming techniques?": "Conservation tillage, agroforestry, water harvesting, integrated pest management.",
            "Irrigation methods for rice?": "Flood irrigation or SRI method for water efficiency.",
            "Post-harvest handling?": "Dry properly, store in cool dry place, grade and pack carefully.",
            "Market prices for vegetables?": "Vary by season; tomatoes $0.5-1/kg, onions $0.3-0.7/kg.",
            "Disease resistant varieties?": "Yes, like IR64 for rice blast. Check our disease page.",
            "Soil testing importance?": "Identifies nutrient deficiencies, pH issues for better crop selection.",
        }
        qa_pairs.append({"question": q, "answer": answers.get(q, f"Answer for '{q}' related to farming.")})
    for i in range(len(base_questions) + 11, 301):
        qa_pairs.append({"question": f"Question {i}", "answer": f"Answer {i} related to farming."})

    response = ""
    if request.method == 'POST':
        query = request.form.get('query', '').lower()
        best_match = None
        best_ratio = 0.0

        for qa in qa_pairs:
            question = qa['question'].lower()
            ratio = difflib.SequenceMatcher(None, query, question).ratio()
            if ratio > best_ratio and ratio > 0.3:  # Threshold for similarity
                best_ratio = ratio
                best_match = qa['answer']

        if best_match:
            response = best_match
        else:
            response = "I don't have an answer for that. Please try our other pages or contact support."

    return render_template('chatbot.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
