from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Optional

# Import User after models are initialized - use lazy import
def _get_user_model():
    try:
        from src.models.user import User
        return User
    except ImportError:
        # Return None if User model is not available yet
        return None

User = None

class RegistrationForm(FlaskForm):
    # Personal Information
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Length(max=120)])
    phone = StringField('Phone Number', validators=[DataRequired(), Length(min=10, max=15)])
    location = StringField('Location', validators=[DataRequired(), Length(min=2, max=100)])

    # Account Information
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=100)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])

    # Role Selection
    user_type = SelectField('Account Type', choices=[('farmer', 'Farmer'), ('buyer', 'Buyer')],
                           validators=[DataRequired()])

    # Farmer-specific fields
    farm_size = FloatField('Farm Size (acres)', validators=[Optional()])
    experience = IntegerField('Experience (years)', validators=[Optional()])
    specialization = StringField('Specialization', validators=[Optional()])

    # Buyer-specific fields
    business_name = StringField('Business Name', validators=[Optional()])
    business_type = SelectField('Business Type',
                               choices=[('', 'Select Business Type'), ('individual', 'Individual'),
                                       ('retailer', 'Retailer'), ('wholesaler', 'Wholesaler'),
                                       ('exporter', 'Exporter'), ('processor', 'Processor')],
                               validators=[Optional()])
    buying_capacity = FloatField('Monthly Buying Capacity (tons)', validators=[Optional()])

    # Terms
    accept_terms = BooleanField('I accept the Terms of Service and Privacy Policy',
                               validators=[DataRequired()])

    def validate_username(self, username):
        User = _get_user_model()
        if User is None:
            # Skip validation if User model is not available yet
            return
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already exists. Please choose a different one.')

    def validate_email(self, email):
        User = _get_user_model()
        if User is None:
            # Skip validation if User model is not available yet
            return
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered. Please use a different email or login instead.')

class LoginForm(FlaskForm):
    username = StringField('Username or Email', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
