"""
Email functionality for the Farmer Assistant application.
"""
from flask import render_template, current_app
from app import mail
from threading import Thread
from flask_mail import Message

def send_async_email(app, msg):
    """Send email asynchronously."""
    with app.app_context():
        mail.send(msg)

def send_email(subject, sender, recipients, text_body, html_body):
    """Send an email with both text and HTML bodies."""
    msg = Message(subject, sender=sender, recipients=recipients)
    msg.body = text_body
    msg.html = html_body
    
    # Send email asynchronously
    Thread(target=send_async_email, args=(current_app._get_current_object(), msg)).start()

def send_password_reset_email(user):
    """Send a password reset email to the user."""
    token = user.get_reset_password_token()
    send_email(
        'Reset Your Password',
        sender=current_app.config['MAIL_DEFAULT_SENDER'],
        recipients=[user.email],
        text_body=render_template('email/reset_password.txt', user=user, token=token),
        html_body=render_template('email/reset_password.html', user=user, token=token)
    )

def send_welcome_email(user):
    """Send a welcome email to new users."""
    send_email(
        'Welcome to Farmer Assistant',
        sender=current_app.config['MAIL_DEFAULT_SENDER'],
        recipients=[user.email],
        text_body=render_template('email/welcome.txt', user=user),
        html_body=render_template('email/welcome.html', user=user)
    )

def send_notification(user, subject, template, **kwargs):
    """Send a notification email to a user."""
    send_email(
        subject,
        sender=current_app.config['MAIL_DEFAULT_SENDER'],
        recipients=[user.email],
        text_body=render_template(f'email/{template}.txt', user=user, **kwargs),
        html_body=render_template(f'email/{template}.html', user=user, **kwargs)
    )
