# 🌾 Farmer Assistant - Smart Farming & Marketplace Platform

Farmer Assistant is an AI-powered web platform designed to empower farmers with intelligent insights, crop recommendations, and a direct marketplace for selling produce. The platform integrates Causal AI for disease prediction and decision support.

## 🌟 Features

### 👨‍🌾 Farmer Features
- **🌱 Crop Recommendation System**
  - Suggests the most suitable crops based on location, soil attributes, and weather data
  - Utilizes machine learning models for data-driven and personalized recommendations

- **📊 Local Market Trends**
  - Displays real-time market price trends for various crops
  - Helps farmers make informed decisions about when and where to sell their produce

- **🧠 Crop Disease Prediction (Causal AI)**
  - Predicts crop diseases from leaf images using AI-based image analysis
  - Provides treatment recommendations and preventive measures

- **💰 Crop Selling Portal**
  - Enables farmers to list crops for sale with price, quantity, and quality details
  - Supports direct communication between buyers and sellers

- **🤝 Farmers Community Forum**
  - Interactive discussion platform for farmers to share knowledge and experiences
  - Promotes collaboration and mutual support within the agricultural community

### 🧑‍💼 Buyer Features
- **🌾 Crop Marketplace**
  - Browse and search for crops listed by farmers
  - Filter by crop type, location, price, and quality

- **📈 Market Trends Dashboard**
  - View price trends and market analytics
  - Make informed purchasing decisions based on historical data

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- PostgreSQL (or SQLite for development)
- Node.js and npm (for frontend assets)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/farmer-assistant.git
   cd farmer-assistant
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with the following content:
   ```env
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=postgresql://username:password@localhost/farmer_assistant
   MAIL_SERVER=smtp.gmail.com
   MAIL_PORT=587
   MAIL_USE_TLS=1
   MAIL_USERNAME=your-email@gmail.com
   MAIL_PASSWORD=your-email-password
   ```

5. **Initialize the database**
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

6. **Run the development server**
   ```bash
   flask run
   ```

7. **Access the application**
   Open your browser and go to `http://127.0.0.1:5000`

## 🛠 Project Structure

```
farmer-assistant/
├── app/
│   ├── __init__.py         # Application factory
│   ├── models/             # Database models
│   ├── routes/             # Application routes
│   ├── static/             # Static files (CSS, JS, images)
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/          # HTML templates
│       ├── auth/           # Authentication templates
│       ├── errors/         # Error pages
│       └── ...
├── migrations/             # Database migrations
├── tests/                  # Test files
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the farmers who provided valuable insights
- Open-source libraries that made this project possible
- The agricultural research community

## 📧 Contact

For any questions or feedback, please contact us at [your-email@example.com](mailto:your-email@example.com)
