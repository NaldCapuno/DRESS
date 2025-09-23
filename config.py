import os

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'dress'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# YOLOv8 Model Configuration
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.5

# Class mapping for the trained model
CLASS_NAMES = {
    0: 'blouse',
    1: 'doll_shoes', 
    2: 'id_student',
    3: 'pants',
    4: 'polo_shirt',
    5: 'shoes',
    6: 'skirt'
}

# Dress code requirements
DRESS_CODE_REQUIREMENTS = {
    'Male': ['polo_shirt', 'pants', 'shoes'],
    'Female': ['blouse', 'skirt', 'shoes']
}

# Class mapping for display names
DISPLAY_NAMES = {
    'blouse': 'Blouse',
    'doll_shoes': 'Shoes',
    'id_student': 'Student ID',
    'pants': 'Black Pants', 
    'polo_shirt': 'Polo Shirt',
    'shoes': 'Shoes',
    'skirt': 'Skirt'
}

# External system integration settings
WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')  # Set via environment variable
WEBHOOK_TIMEOUT = 10  # seconds
WEBHOOK_RETRY_ATTEMPTS = 3

# Message queue settings (Redis)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_QUEUE_NAME = 'dress_violations'

# Violation logging settings
VIOLATION_LOG_RETENTION_DAYS = 30
VIOLATION_THROTTLE_SECONDS = 5  # Prevent spam for same violation
