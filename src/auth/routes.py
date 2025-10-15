"""
Authentication routes for the Farmer Assistant application.
Handles user registration, login, logout, and account management.
"""
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import db
from app.models.user import User, Farmer, Buyer
from app.auth.forms import LoginForm, RegistrationForm

bp = Blueprint('auth', __name__)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('auth.login'))

        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.index')
        return redirect(next_page)

    return render_template('auth/login.html', title='Sign In', form=form)

@bp.route('/logout')
def logout():
    """Handle user logout."""
    logout_user()
    return redirect(url_for('main.index'))

@bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle new user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        if form.user_type.data == 'farmer':
            user = Farmer(
                username=form.username.data,
                email=form.email.data,
                phone=form.phone.data,
                farm_name=form.farm_name.data,
                farm_location=form.farm_location.data,
                farm_size=form.farm_size.data,
                primary_crops=form.primary_crops.data,
                full_name=form.full_name.data,
                address=form.address.data,
                city=form.city.data,
                state=form.state.data,
                country=form.country.data,
                postal_code=form.postal_code.data
            )
        else:
            user = Buyer(
                username=form.username.data,
                email=form.email.data,
                phone=form.phone.data,
                business_name=form.business_name.data,
                business_type=form.business_type.data,
                full_name=form.full_name.data,
                address=form.address.data,
                city=form.city.data,
                state=form.state.data,
                country=form.country.data,
                postal_code=form.postal_code.data
            )

        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()

        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html', title='Register', form=form)
