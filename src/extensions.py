"""
Flask extensions for the Farmer Assistant application.
"""
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_migrate import Migrate
from flask_caching import Cache
from flask_babel import Babel

# Initialize extensions
db = SQLAlchemy()
mail = Mail()
migrate = Migrate()
cache = Cache()
babel = Babel()

# This will be imported by app/__init__.py
