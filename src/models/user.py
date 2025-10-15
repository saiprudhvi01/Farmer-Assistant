"""
User models for the Farmer Assistant application.
"""
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Import db later to avoid circular imports
db = None

def set_db(db_instance):
    """Set the database instance after app creation."""
    global db
    db = db_instance

    # Now define the models after db is available
    class User(UserMixin, db.Model):
        """Base user model."""
        __tablename__ = 'users'

        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(128), nullable=False)
        phone = db.Column(db.String(20))
        user_type = db.Column(db.String(20), nullable=False)  # 'farmer'
        is_active = db.Column(db.Boolean, default=True)
        created_at = db.Column(db.DateTime, default=db.func.now())

        # Farmer-specific fields
        farm_name = db.Column(db.String(100))
        farm_location = db.Column(db.String(100))
        farm_size = db.Column(db.Float)  # in acres/hectares
        primary_crops = db.Column(db.String(200))  # comma-separated

        # Profile fields
        full_name = db.Column(db.String(100))
        address = db.Column(db.Text)
        city = db.Column(db.String(50))
        state = db.Column(db.String(50))
        country = db.Column(db.String(50), default='India')
        postal_code = db.Column(db.String(20))

        # Polymorphic discriminator
        __mapper_args__ = {'polymorphic_on': user_type}

        # Relationships (defined on base class since all user types can have these)
        crops = db.relationship('CropListing', back_populates='farmer', lazy='dynamic')
        posts = db.relationship('ForumPost', back_populates='author', lazy='dynamic')
        purchases = db.relationship('Purchase', back_populates='buyer', lazy='dynamic')

        def set_password(self, password):
            """Set password hash."""
            self.password_hash = generate_password_hash(password)

        def check_password(self, password):
            """Check password hash."""
            return check_password_hash(self.password_hash, password)

        def get_reset_password_token(self, expires_in=600):
            """Generate password reset token."""
            from itsdangerous import URLSafeTimedSerializer
            import current_app

            s = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
            return s.dumps({'user_id': self.id}).decode('utf-8')

        @staticmethod
        def verify_reset_password_token(token):
            """Verify password reset token."""
            from itsdangerous import URLSafeTimedSerializer
            import current_app

            s = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
            try:
                user_id = s.loads(token.encode('utf-8'))['user_id']
            except:
                return None

            return User.query.get(user_id)

        def get_full_name(self):
            """Get the user's full name."""
            return self.full_name or f"{self.username}"

        def is_farmer(self):
            """Check if user is a farmer."""
            return self.user_type == 'farmer'

        def is_buyer(self):
            """Check if user is a buyer."""
            return self.user_type == 'buyer'

    class Farmer(User):
        """Farmer-specific model."""
        __tablename__ = 'farmers'

        id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)

        # Polymorphic identity for inheritance
        __mapper_args__ = {'polymorphic_identity': 'farmer'}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.user_type = 'farmer'

    # Marketplace models
    class CropListing(db.Model):
        """Model for crops listed by farmers."""
        __tablename__ = 'crop_listings'

        id = db.Column(db.Integer, primary_key=True)
        farmer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        crop_name = db.Column(db.String(50), nullable=False)
        variety = db.Column(db.String(50))
        quantity = db.Column(db.Float, nullable=False)
        unit = db.Column(db.String(20), nullable=False)  # kg, quintal, ton, etc.
        price_per_unit = db.Column(db.Float, nullable=False)
        location = db.Column(db.String(100), nullable=False)
        description = db.Column(db.Text)
        is_organic = db.Column(db.Boolean, default=False)
        is_available = db.Column(db.Boolean, default=True)
        images = db.Column(db.Text)  # JSON array of image URLs
        harvest_date = db.Column(db.Date)
        expiry_date = db.Column(db.Date)
        created_at = db.Column(db.DateTime, default=db.func.now())
        updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

        # Relationships
        farmer = db.relationship('User', back_populates='crops')
        offers = db.relationship('Offer', back_populates='listing', lazy='dynamic')

    class Offer(db.Model):
        """Model for offers/purchases on crop listings."""
        __tablename__ = 'offers'

        id = db.Column(db.Integer, primary_key=True)
        listing_id = db.Column(db.Integer, db.ForeignKey('crop_listings.id'), nullable=False)
        buyer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        offer_price = db.Column(db.Float, nullable=False)
        offer_quantity = db.Column(db.Float, nullable=False)
        message = db.Column(db.Text)
        status = db.Column(db.String(20), default='pending')  # pending, accepted, rejected, completed
        created_at = db.Column(db.DateTime, default=db.func.now())
        updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

        # Relationships
        listing = db.relationship('CropListing', back_populates='offers')
        buyer = db.relationship('User', foreign_keys=[buyer_id])

    # Community models
    class ForumPost(db.Model):
        """Model for forum posts."""
        __tablename__ = 'forum_posts'

        id = db.Column(db.Integer, primary_key=True)
        author_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        title = db.Column(db.String(200), nullable=False)
        content = db.Column(db.Text, nullable=False)
        category = db.Column(db.String(50), default='general')
        tags = db.Column(db.String(200))  # comma-separated tags
        likes_count = db.Column(db.Integer, default=0)
        views_count = db.Column(db.Integer, default=0)
        is_pinned = db.Column(db.Boolean, default=False)
        created_at = db.Column(db.DateTime, default=db.func.now())
        updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

        # Relationships
        author = db.relationship('User', back_populates='posts')
        comments = db.relationship('ForumComment', back_populates='post', lazy='dynamic')
        likes = db.relationship('PostLike', back_populates='post', lazy='dynamic')

    class ForumComment(db.Model):
        """Model for forum comments."""
        __tablename__ = 'forum_comments'

        id = db.Column(db.Integer, primary_key=True)
        post_id = db.Column(db.Integer, db.ForeignKey('forum_posts.id'), nullable=False)
        author_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        content = db.Column(db.Text, nullable=False)
        parent_id = db.Column(db.Integer, db.ForeignKey('forum_comments.id'))  # for nested comments
        created_at = db.Column(db.DateTime, default=db.func.now())
        updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

        # Relationships
        post = db.relationship('ForumPost', back_populates='comments')
        author = db.relationship('User')
        parent = db.relationship('ForumComment', remote_side=[id])
        replies = db.relationship('ForumComment', back_populates='parent', lazy='dynamic')

    class PostLike(db.Model):
        """Model for post likes."""
        __tablename__ = 'post_likes'

        id = db.Column(db.Integer, primary_key=True)
        post_id = db.Column(db.Integer, db.ForeignKey('forum_posts.id'), nullable=False)
        user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        created_at = db.Column(db.DateTime, default=db.func.now())

        # Relationships
        post = db.relationship('ForumPost', back_populates='likes')
        user = db.relationship('User')

        # Ensure unique like per user per post
        __table_args__ = (db.UniqueConstraint('post_id', 'user_id', name='unique_post_like'),)

    # Purchase models
    class Purchase(db.Model):
        """Model for completed purchases."""
        __tablename__ = 'purchases'

        id = db.Column(db.Integer, primary_key=True)
        buyer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        listing_id = db.Column(db.Integer, db.ForeignKey('crop_listings.id'), nullable=False)
        quantity = db.Column(db.Float, nullable=False)
        unit_price = db.Column(db.Float, nullable=False)
        total_amount = db.Column(db.Float, nullable=False)
        status = db.Column(db.String(20), default='pending')  # pending, completed, cancelled
        payment_status = db.Column(db.String(20), default='pending')  # pending, paid, refunded
        delivery_status = db.Column(db.String(20), default='pending')  # pending, shipped, delivered
        created_at = db.Column(db.DateTime, default=db.func.now())
        completed_at = db.Column(db.DateTime)

        # Relationships
        buyer = db.relationship('User', back_populates='purchases')
        listing = db.relationship('CropListing')

    # Make models available at module level
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, 'User', User)
    setattr(current_module, 'Farmer', Farmer)
    # setattr(current_module, 'Buyer', Buyer)  # Removed buyer
    setattr(current_module, 'CropListing', CropListing)
    setattr(current_module, 'Offer', Offer)
    setattr(current_module, 'ForumPost', ForumPost)
    setattr(current_module, 'ForumComment', ForumComment)
    setattr(current_module, 'PostLike', PostLike)
    setattr(current_module, 'Purchase', Purchase)

    # Also update the parent models module
    try:
        models_module = sys.modules.get('src.models')
        if models_module:
            setattr(models_module, 'User', User)
            setattr(models_module, 'Farmer', Farmer)
            # setattr(models_module, 'Buyer', Buyer)  # Removed buyer
            setattr(models_module, 'CropListing', CropListing)
            setattr(models_module, 'Offer', Offer)
            setattr(models_module, 'ForumPost', ForumPost)
            setattr(models_module, 'ForumComment', ForumComment)
            setattr(models_module, 'PostLike', PostLike)
            setattr(models_module, 'Purchase', Purchase)
    except:
        pass
