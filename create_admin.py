#!/usr/bin/env python3
"""
Script to create a new admin user for the DRESS system.
Run this script to add new admin users to the database.
"""

import sys
import os
import pymysql
from werkzeug.security import generate_password_hash

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Database configuration (same as in app.py)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'database': os.getenv('DB_NAME', 'dress'),
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True
}

def create_admin():
    """Create a new admin user interactively"""
    print("=== DRESS System - Create Admin User ===\n")
    
    # Get admin details
    full_name = input("Enter full name: ").strip()
    if not full_name:
        print("Error: Full name is required")
        return False
    
    email = input("Enter email address: ").strip()
    if not email:
        print("Error: Email is required")
        return False
    
    # Validate email format (basic check)
    if '@' not in email or '.' not in email:
        print("Error: Please enter a valid email address")
        return False
    
    print("\nAvailable roles:")
    print("1. Security")
    print("2. OSAS") 
    print("3. Dean")
    print("4. Guidance")
    
    role_choice = input("Select role (1-4): ").strip()
    role_map = {
        '1': 'Security',
        '2': 'OSAS', 
        '3': 'Dean',
        '4': 'Guidance'
    }
    
    if role_choice not in role_map:
        print("Error: Invalid role selection")
        return False
    
    role = role_map[role_choice]
    
    password = input("Enter password: ").strip()
    if not password:
        print("Error: Password is required")
        return False
    
    if len(password) < 6:
        print("Error: Password must be at least 6 characters long")
        return False
    
    confirm_password = input("Confirm password: ").strip()
    if password != confirm_password:
        print("Error: Passwords do not match")
        return False
    
    # Hash the password
    password_hash = generate_password_hash(password)
    
    try:
        # Connect to database
        conn = pymysql.connect(**DB_CONFIG)
        
        try:
            with conn.cursor() as cur:
                # Check if email already exists
                cur.execute("SELECT admin_id FROM Admins WHERE email = %s", (email,))
                if cur.fetchone():
                    print(f"Error: Admin with email '{email}' already exists")
                    return False
                
                # Insert new admin
                cur.execute(
                    """
                    INSERT INTO Admins (full_name, email, password_hash, role)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (full_name, email, password_hash, role)
                )
                
                admin_id = cur.lastrowid
                print(f"\n✅ Admin user created successfully!")
                print(f"Admin ID: {admin_id}")
                print(f"Name: {full_name}")
                print(f"Email: {email}")
                print(f"Role: {role}")
                print(f"\nThe admin can now login at: http://localhost:5000/login")
                
                return True
                
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error creating admin: {e}")
        return False

def list_admins():
    """List all existing admin users"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT admin_id, full_name, email, role, created_at, last_login
                    FROM Admins
                    ORDER BY created_at DESC
                """)
                
                admins = cur.fetchall()
                
                if not admins:
                    print("No admin users found.")
                    return
                
                print("\n=== Existing Admin Users ===")
                print(f"{'ID':<5} {'Name':<20} {'Email':<30} {'Role':<10} {'Created':<20} {'Last Login':<20}")
                print("-" * 100)
                
                for admin in admins:
                    last_login = admin['last_login'].strftime('%Y-%m-%d %H:%M') if admin['last_login'] else 'Never'
                    created = admin['created_at'].strftime('%Y-%m-%d %H:%M')
                    
                    print(f"{admin['admin_id']:<5} {admin['full_name']:<20} {admin['email']:<30} {admin['role']:<10} {created:<20} {last_login:<20}")
                    
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error listing admins: {e}")

if __name__ == "__main__":
    print("DRESS System - Admin Management")
    print("1. Create new admin")
    print("2. List existing admins")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == '1':
        create_admin()
    elif choice == '2':
        list_admins()
    else:
        print("Invalid option")
        sys.exit(1)
