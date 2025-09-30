"""
Database module for the DRESS application.
Handles all database connections and operations.
"""

import os
import pymysql
import logging
from typing import Optional, Dict, Any
from werkzeug.security import check_password_hash


class DatabaseConfig:
    """Database configuration class."""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '3306'))
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', 'root')
        self.database = os.getenv('DB_NAME', 'dress')


class DatabaseManager:
    """Database manager class for handling connections and operations."""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Get a new database connection."""
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
    
    # ------------------ Admin Authentication ------------------
    def get_admin_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch an admin record by username."""
        if not username:
            return None
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT admin_id, username, password_hash, role, created_at
                        FROM admins
                        WHERE username = %s
                        LIMIT 1
                        """,
                        (username,)
                    )
                    return cur.fetchone()
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB admin fetch error: {e}")
            return None

    def verify_admin_credentials(self, username: str, password: str, required_role: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Verify admin username/password and optional role. Returns admin dict without password on success."""
        admin = self.get_admin_by_username(username)
        if not admin:
            return None
        try:
            if not check_password_hash(admin.get('password_hash', ''), password or ''):
                return None
            if required_role and str(admin.get('role') or '').lower() != str(required_role or '').lower():
                return None
            # Remove sensitive field before returning
            sanitized = dict(admin)
            sanitized.pop('password_hash', None)
            return sanitized
        except Exception as e:
            self.logger.error(f"Password check error: {e}")
            return None

    def find_student_by_rfid(self, rfid_uid: str) -> Optional[Dict[str, Any]]:
        """Find a student by their RFID UID."""
        if not rfid_uid:
            return None
        
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT student_id, rfid_uid, name, gender, year_level, course, college, photo
                        FROM students
                        WHERE rfid_uid = %s
                        LIMIT 1
                        """,
                        (rfid_uid,)
                    )
                    row = cur.fetchone()
                    return row
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB lookup error: {e}")
            return None
    
    def insert_rfid_log(self, rfid_uid: str, student_id: Optional[int], status: str) -> bool:
        """Insert an RFID log entry."""
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO rfid_logs (student_id, rfid_uid, status)
                        VALUES (%s, %s, %s)
                        """,
                        (student_id, rfid_uid, status)
                    )
                    return True
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB log insert error: {e}")
            return False
    
    def insert_violation(self, student_id: Optional[str], violation_type: str, 
                        image_proof_rel_path: Optional[str], 
                        recorded_by: Optional[int] = None) -> Optional[int]:
        """Insert a violation record."""
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO violations (student_id, recorded_by, violation_type, image_proof)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (student_id, recorded_by, violation_type, image_proof_rel_path)
                    )
                    return cur.lastrowid
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB violation insert error: {e}")
            return None
    
    def lookup_and_log(self, rfid_uid: str) -> Dict[str, Any]:
        """Look up a student by RFID and log the attempt."""
        student = self.find_student_by_rfid(rfid_uid)
        if student:
            self.insert_rfid_log(rfid_uid, student.get('student_id'), 'valid')
            return {
                'matched': True,
                'student_id': student.get('student_id'),
                'name': student.get('name'),
                'rfid_uid': student.get('rfid_uid'),
                'gender': student.get('gender'),
                'year_level': student.get('year_level'),
                'course': student.get('course'),
                'college': student.get('college'),
                'photo': student.get('photo'),
            }
        else:
            self.insert_rfid_log(rfid_uid, None, 'unregistered')
            return {'matched': False}


# Global database manager instance
db_manager = DatabaseManager()
