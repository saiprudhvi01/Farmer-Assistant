"""
Forms for user authentication and account management.
"""
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, TextAreaField, FloatField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError, Length, Regexp, Optional
from app.models.user import User

class LoginForm(FlaskForm):
    """Form for user login."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    """Form for new user registration."""
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=4, max=25),
        Regexp('^[A-Za-z0-9_]+$', message='Username can only contain letters, numbers, and underscores')
    ])

    email = StringField('Email', validators=[
        DataRequired(),
        Email(),
        Length(max=120)
    ])

    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters long')
    ])

    password2 = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])

    user_type = SelectField('I am a', choices=[
        ('farmer', 'Farmer'),
        ('buyer', 'Buyer')
    ], validators=[DataRequired()])

    # Common fields
    full_name = StringField('Full Name', validators=[DataRequired(), Length(max=100)])
    phone = StringField('Phone Number', validators=[
        DataRequired(),
        Regexp(r'^\+?[\d\s\-\(\)]+$', message='Please enter a valid phone number')
    ])

    # Farmer-specific fields
    farm_name = StringField('Farm Name', validators=[Optional(), Length(max=100)])
    farm_location = StringField('Farm Location', validators=[Optional(), Length(max=100)])
    farm_size = FloatField('Farm Size (acres)', validators=[Optional()])
    primary_crops = StringField('Primary Crops', validators=[Optional(), Length(max=200)])

    # Buyer-specific fields
    business_name = StringField('Business Name', validators=[Optional(), Length(max=100)])
    business_type = SelectField('Business Type', choices=[
        ('', 'Select business type'),
        ('wholesaler', 'Wholesaler'),
        ('retailer', 'Retailer'),
        ('processor', 'Processor'),
        ('exporter', 'Exporter'),
        ('individual', 'Individual Buyer'),
        ('other', 'Other')
    ], validators=[Optional()])

    # Address fields
    address = TextAreaField('Address', validators=[Optional(), Length(max=500)])
    city = StringField('City', validators=[Optional(), Length(max=50)])
    state = StringField('State', validators=[Optional(), Length(max=50)])
    country = StringField('Country', validators=[Optional(), Length(max=50)])
    postal_code = StringField('Postal Code', validators=[Optional(), Length(max=20)])

    def validate_username(self, username):
        """Validate that username is unique."""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('This username is already taken. Please choose a different one.')

    def validate_email(self, email):
        """Validate that email is unique."""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('This email is already registered. Please use a different email or login instead.')
