"""
Farmer Assistant - Smart Farming & Marketplace Platform

This module initializes the Flask application and registers blueprints.
"""
from flask import Flask
from config import Config

def create_app(config_class=Config):
    """Application factory function to create and configure the Flask app."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    from app.extensions import db, login_manager, mail
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)

    # Register blueprints
    from app.auth import bp as auth_bp
    from app.main import bp as main_bp
    from app.marketplace import bp as marketplace_bp
    from app.crop_recommendation import bp as crop_rec_bp
    from app.disease_prediction import bp as disease_pred_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(main_bp)
    app.register_blueprint(marketplace_bp, url_prefix='/marketplace')
    app.register_blueprint(crop_rec_bp, url_prefix='/crop-recommendation')
    app.register_blueprint(disease_pred_bp, url_prefix='/disease-prediction')

    # Create database tables
    with app.app_context():
        db.create_all()

    return app
