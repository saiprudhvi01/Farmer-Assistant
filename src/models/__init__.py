"""
Database models for the Farmer Assistant application.
"""
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from src.extensions import db

# Import user module to access set_db function
from . import user

# Models will be available after set_db() is called in app.py
# For now, define placeholder classes that will be replaced
User = None
Farmer = None
Buyer = None
CropListing = None
Offer = None
ForumPost = None
ForumComment = None
PostLike = None
Purchase = None

# Make all models available at package level
__all__ = ['User', 'Farmer', 'Buyer', 'CropListing', 'Offer', 'ForumPost', 'ForumComment', 'PostLike', 'Purchase']
