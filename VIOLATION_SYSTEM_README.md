# Dress Code Violation Detection System

This document explains how the violation detection system works and how to integrate it with external systems.

## Overview

The system automatically detects dress code violations using YOLOv8 object detection and processes them through multiple channels:
- Database logging
- Webhook notifications
- Message queue publishing
- API endpoints for external systems

## Violation Detection Logic

### 1. YOLOv8 Class Detection
The system detects these clothing items:
```python
CLASS_NAMES = {
    0: 'blouse',      # Female top
    1: 'doll_shoes',  # Female shoes  
    2: 'id_student',  # Student ID
    3: 'pants',       # Male bottom
    4: 'polo_shirt',  # Male top
    5: 'shoes',       # Male shoes
    6: 'skirt'        # Female bottom
}
```

### 2. Gender Detection
Gender is determined by analyzing detected clothing items:
- **Female items**: blouse, skirt, doll_shoes
- **Male items**: polo_shirt, pants, shoes
- **Default**: Male (if unclear)

### 3. Dress Code Requirements
```python
DRESS_CODE_REQUIREMENTS = {
    'Male': ['polo_shirt', 'pants', 'shoes'],
    'Female': ['blouse', 'skirt', 'shoes']
}
```

### 4. Violation Triggering
A violation occurs when:
- Required items for the detected gender are missing
- Confidence threshold (0.5) is met for detected items
- Not throttled (prevents spam - 5 second cooldown)

## Integration Methods

### 1. API Endpoints

#### Check Violations
```bash
POST /api/violation-check
Content-Type: application/json

{
    "detected_classes": ["pants", "shoes"],
    "student_id": "STU001",
    "location": "Entrance Gate"
}
```

Response:
```json
{
    "success": true,
    "has_violation": true,
    "violation_details": {
        "missing_items": ["Polo Shirt"],
        "detected_items": ["Black Pants", "Shoes"],
        "gender": "Male",
        "student_id": "STU001",
        "location": "Entrance Gate"
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

#### Get Violations
```bash
GET /api/violations?student_id=STU001&status=Pending&limit=10&offset=0
```

#### Update Violation Status
```bash
PUT /api/violations/123/status
Content-Type: application/json

{
    "status": "Resolved"
}
```

#### Health Check
```bash
GET /api/health
```

### 2. Webhook Integration

Configure webhook URL in environment variables:
```bash
export WEBHOOK_URL="https://your-system.com/violations"
```

Webhook payload format:
```json
{
    "student_id": "STU001",
    "missing_items": ["Polo Shirt"],
    "detected_items": ["Black Pants", "Shoes"],
    "gender": "Male",
    "location": "Live Camera",
    "timestamp": "2024-01-15T10:30:00",
    "violation_type": "dress_code"
}
```

### 3. Message Queue (Redis)

Configure Redis connection:
```bash
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"
```

Messages are published to the `dress_violations` queue with this format:
```json
{
    "type": "dress_code_violation",
    "data": {
        "student_id": "STU001",
        "missing_items": ["Polo Shirt"],
        "detected_items": ["Black Pants", "Shoes"],
        "gender": "Male",
        "location": "Live Camera"
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

## Database Schema

### Violations Table
```sql
CREATE TABLE `violations` (
  `violation_id` int NOT NULL AUTO_INCREMENT,
  `student_id` varchar(20) DEFAULT NULL,
  `missing_item` varchar(100) NOT NULL,
  `detected_items` text DEFAULT NULL,
  `gender` enum('Male','Female') DEFAULT NULL,
  `location` varchar(100) DEFAULT 'Web Detection',
  `status` enum('Pending','Resolved','Dismissed') DEFAULT 'Pending',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`violation_id`)
);
```

### Useful Queries

#### Get violation statistics:
```sql
CALL GetViolationStats(30); -- Last 30 days
```

#### Get violation summary:
```sql
SELECT * FROM violation_summary WHERE status = 'Pending';
```

## Configuration

### Environment Variables
```bash
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=root
DB_NAME=dress

# Webhook
WEBHOOK_URL=https://your-system.com/violations

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Config File (config.py)
```python
# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# Violation throttling (seconds)
VIOLATION_THROTTLE_SECONDS = 5

# Webhook settings
WEBHOOK_TIMEOUT = 10
WEBHOOK_RETRY_ATTEMPTS = 3
```

## Usage Examples

### 1. Python Integration
```python
import requests

# Check for violations
response = requests.post('http://localhost:5000/api/violation-check', json={
    'detected_classes': ['pants', 'shoes'],
    'student_id': 'STU001',
    'location': 'Entrance'
})

violation_data = response.json()
if violation_data['has_violation']:
    print(f"Violation detected: {violation_data['violation_details']['missing_items']}")
```

### 2. Webhook Receiver (Flask)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/violations', methods=['POST'])
def handle_violation():
    data = request.json
    
    # Process violation
    student_id = data.get('student_id')
    missing_items = data.get('missing_items')
    
    # Your violation handling logic here
    print(f"Violation for student {student_id}: {missing_items}")
    
    return jsonify({'status': 'received'})
```

### 3. Redis Consumer
```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

while True:
    # Block until message available
    message = r.brpop('dress_violations', timeout=1)
    if message:
        violation_data = json.loads(message[1])
        print(f"Received violation: {violation_data}")
```

## Error Handling

The system includes comprehensive error handling:
- Database connection failures
- Webhook delivery failures (with retry)
- Redis connection issues
- Invalid API requests

All errors are logged with appropriate severity levels.

## Monitoring

### Health Check Endpoint
```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
    "status": "healthy",
    "database": "healthy",
    "model": "loaded",
    "timestamp": "2024-01-15T10:30:00"
}
```

### Logs
Monitor application logs for:
- Violation detections
- Webhook delivery status
- Database errors
- API usage

## Security Considerations

1. **API Authentication**: Consider adding API keys for external systems
2. **Webhook Security**: Use HTTPS and verify webhook signatures
3. **Database Security**: Use proper user permissions
4. **Input Validation**: All inputs are validated before processing

## Troubleshooting

### Common Issues

1. **Violations not being detected**
   - Check confidence threshold
   - Verify YOLOv8 model is loaded
   - Check class mappings

2. **Webhook not working**
   - Verify WEBHOOK_URL is set
   - Check network connectivity
   - Review webhook endpoint logs

3. **Database errors**
   - Verify database connection
   - Check table schema
   - Review foreign key constraints

### Debug Mode
Enable debug logging by setting Flask debug mode:
```python
app.run(debug=True)
```

## Support

For issues or questions:
1. Check application logs
2. Verify configuration
3. Test API endpoints
4. Review database schema
