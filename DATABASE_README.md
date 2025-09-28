# Database Module Documentation

## Overview

The database functionality has been separated into a dedicated module (`database.py`) for better organization and maintainability.

## Structure

### `database.py`

Contains two main classes:

#### `DatabaseConfig`
- Handles database configuration
- Reads configuration from environment variables with sensible defaults
- Configurable parameters:
  - `DB_HOST` (default: localhost)
  - `DB_PORT` (default: 3306)
  - `DB_USER` (default: root)
  - `DB_PASSWORD` (default: root)
  - `DB_NAME` (default: dress)

#### `DatabaseManager`
- Main class for database operations
- Methods:
  - `get_connection()`: Creates a new database connection
  - `find_student_by_rfid(rfid_uid)`: Finds a student by RFID UID
  - `insert_rfid_log(rfid_uid, student_id, status)`: Logs RFID scan attempts
  - `insert_violation(student_id, violation_type, image_proof_rel_path, recorded_by)`: Records violations
  - `lookup_and_log(rfid_uid)`: Combined lookup and logging operation

## Usage

### In `app.py`
```python
from database import db_manager

# Find a student
student = db_manager.find_student_by_rfid("ABC123")

# Insert a violation
violation_id = db_manager.insert_violation(
    student_id="12345",
    violation_type="Incomplete uniform",
    image_proof_rel_path="static/captures/violation.jpg"
)

# Lookup and log RFID scan
result = db_manager.lookup_and_log("ABC123")
```

## Benefits

1. **Separation of Concerns**: Database logic is isolated from application logic
2. **Reusability**: Database operations can be easily reused across different parts of the application
3. **Maintainability**: Database-related changes only need to be made in one place
4. **Testability**: Database operations can be easily unit tested
5. **Configuration Management**: Database settings are centralized and environment-configurable

## Environment Variables

Set these environment variables to override default database settings:

```bash
export DB_HOST=your_host
export DB_PORT=3306
export DB_USER=your_username
export DB_PASSWORD=your_password
export DB_NAME=your_database
```

## Dependencies

- `pymysql`: MySQL database connector
- `logging`: For error logging
- `os`: For environment variable access
- `typing`: For type hints
