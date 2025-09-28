# DRESS System - Login Setup Guide

This guide explains how to set up and use the login system for the DRESS (Dress Code Enforcement System).

## Features

- **Secure Authentication**: Password hashing using Werkzeug
- **Role-based Access**: Support for Security, OSAS, Dean, and Guidance roles
- **Session Management**: Secure session handling with Flask sessions
- **Modern UI**: Clean, responsive login interface
- **Admin Dashboard**: Protected dashboard with user information

## Setup Instructions

### 1. Database Migration

First, update your database schema to match the new Admins table:

```bash
# Run the migration script
mysql -u root -p dress < migrate_admins_table.sql
```

This will:
- Drop the existing `admins` table
- Create the new `Admins` table with the correct schema
- Insert sample admin users

### 2. Sample Admin Users

The migration script creates 4 sample admin users:

| Role | Email | Password |
|------|-------|----------|
| Security | security@university.edu | admin123 |
| OSAS | osas@university.edu | admin123 |
| Dean | dean@university.edu | admin123 |
| Guidance | guidance@university.edu | admin123 |

**⚠️ Important**: Change these default passwords after first login!

### 3. Create Additional Admin Users

Use the provided script to create new admin users:

```bash
python create_admin.py
```

This interactive script will:
- Prompt for admin details (name, email, role, password)
- Hash the password securely
- Insert the new admin into the database
- Validate email format and password strength

### 4. Start the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Login Process

1. Navigate to `http://localhost:5000`
2. You'll be redirected to the login page
3. Enter your email, password, and select your role
4. Click "Login"
5. Upon successful login, you'll be redirected to the dashboard

### Dashboard Features

- **Admin Information**: Shows logged-in admin's name and role
- **Logout**: Secure logout that clears the session
- **Protected Routes**: All dashboard features require authentication

### Security Features

- **Password Hashing**: All passwords are hashed using Werkzeug's secure hashing
- **Session Security**: Flask sessions with secret key
- **Role Validation**: Users can only access with their assigned role
- **Login Tracking**: Last login timestamp is recorded

## File Structure

```
DRESS/
├── templates/
│   ├── login.html          # Login page template
│   └── dashboard.html      # Updated dashboard with admin info
├── static/
│   └── style.css           # Updated styles for navbar
├── app.py                 # Updated Flask app with auth routes
├── migrate_admins_table.sql # Database migration script
├── create_admin.py         # Admin creation utility
└── LOGIN_SETUP.md         # This guide
```

## API Endpoints

### Authentication Routes

- `GET /` - Redirects to login or dashboard
- `GET /login` - Login page
- `POST /login` - Authenticate user
- `GET /logout` - Logout and clear session
- `GET /dashboard` - Protected dashboard (requires login)

### Login API

**POST /login**
```json
{
    "email": "admin@university.edu",
    "password": "password123",
    "role": "Security"
}
```

**Response (Success):**
```json
{
    "success": true,
    "message": "Login successful",
    "admin": {
        "name": "John Doe",
        "email": "admin@university.edu",
        "role": "Security"
    }
}
```

**Response (Error):**
```json
{
    "success": false,
    "error": "Invalid email or password"
}
```

## Database Schema

The `Admins` table structure:

```sql
CREATE TABLE `Admins` (
    `admin_id` INT PRIMARY KEY AUTO_INCREMENT,
    `full_name` VARCHAR(100) NOT NULL,
    `email` VARCHAR(100) UNIQUE NOT NULL,
    `password_hash` VARCHAR(255) NOT NULL,
    `role` ENUM('Security', 'OSAS', 'Dean', 'Guidance') NOT NULL,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `last_login` TIMESTAMP NULL
);
```

## Troubleshooting

### Common Issues

1. **"Invalid email or password"**
   - Check email spelling
   - Verify password is correct
   - Ensure role matches the account

2. **"Access denied"**
   - Make sure you selected the correct role for your account

3. **Database connection errors**
   - Verify database credentials in environment variables
   - Ensure the `dress` database exists
   - Check that the `Admins` table was created successfully

4. **Session issues**
   - Clear browser cookies and try again
   - Restart the Flask application

### Environment Variables

Make sure these are set correctly:

```bash
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=dress
SECRET_KEY=your-secret-key-change-this-in-production
```

## Security Recommendations

1. **Change Default Passwords**: Immediately change the default passwords for sample users
2. **Strong Secret Key**: Use a strong, random secret key for production
3. **HTTPS**: Use HTTPS in production for secure session transmission
4. **Regular Updates**: Keep dependencies updated for security patches
5. **Access Logs**: Monitor login attempts and failed authentications

## Support

For issues or questions about the login system, check:
1. Application logs for error messages
2. Database connectivity
3. Browser console for JavaScript errors
4. Network tab for API request/response details
