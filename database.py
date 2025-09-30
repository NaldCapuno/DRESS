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

    # ------------------ OSAS Queries ------------------
    def get_violations(self, start_dt: Optional[str] = None, end_dt: Optional[str] = None,
                       academic_year: Optional[str] = None, semester: Optional[str] = None,
                       group_by: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List violations with optional date filters and pagination. Returns rows and total count."""
        try:
            conn = self.get_connection()
            try:
                where = []
                params = []
                # Date range filter
                if start_dt:
                    where.append("v.timestamp >= %s")
                    params.append(start_dt)
                if end_dt:
                    where.append("v.timestamp <= %s")
                    params.append(end_dt)
                # Academic year/semester derived filter if provided (best-effort)
                # Expect academic_year like "2024-2025" and semester in {"1","2"}
                # We translate to approximate date ranges: Sem1 Aug-Dec, Sem2 Jan-May
                if academic_year and semester in {"1", "2"}:
                    try:
                        start_year = int(academic_year.split('-')[0])
                        if semester == "1":
                            ay_start = f"{start_year}-08-01 00:00:00"
                            ay_end = f"{start_year}-12-31 23:59:59"
                        else:
                            ay_start = f"{start_year+1}-01-01 00:00:00"
                            ay_end = f"{start_year+1}-05-31 23:59:59"
                        where.append("v.timestamp BETWEEN %s AND %s")
                        params.extend([ay_start, ay_end])
                    except Exception:
                        pass

                where_sql = (" WHERE " + " AND ".join(where)) if where else ""

                base_select = (
                    "SELECT v.violation_id, v.student_id, v.recorded_by, v.violation_type, v.timestamp, v.image_proof, v.status, "
                    "s.name, s.gender, s.course, s.college "
                    "FROM violations v LEFT JOIN students s ON v.student_id = s.student_id"
                )

                with conn.cursor() as cur:
                    # Total count
                    cur.execute(f"SELECT COUNT(*) AS cnt FROM violations v{where_sql}", params)
                    total = (cur.fetchone() or {}).get('cnt', 0)

                    # Page
                    cur.execute(
                        f"{base_select}{where_sql} ORDER BY v.timestamp DESC LIMIT %s OFFSET %s",
                        params + [int(limit), int(offset)]
                    )
                    rows = cur.fetchall() or []
                    return {"rows": rows, "total": total}
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB get_violations error: {e}")
            return {"rows": [], "total": 0}

    def get_violations_analytics(self, start_dt: Optional[str] = None, end_dt: Optional[str] = None,
                                  academic_year: Optional[str] = None, semester: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregate analytics for violations."""
        try:
            conn = self.get_connection()
            try:
                where = []
                params = []
                if start_dt:
                    where.append("v.timestamp >= %s")
                    params.append(start_dt)
                if end_dt:
                    where.append("v.timestamp <= %s")
                    params.append(end_dt)
                if academic_year and semester in {"1", "2"}:
                    try:
                        start_year = int(academic_year.split('-')[0])
                        if semester == "1":
                            ay_start = f"{start_year}-08-01 00:00:00"
                            ay_end = f"{start_year}-12-31 23:59:59"
                        else:
                            ay_start = f"{start_year+1}-01-01 00:00:00"
                            ay_end = f"{start_year+1}-05-31 23:59:59"
                        where.append("v.timestamp BETWEEN %s AND %s")
                        params.extend([ay_start, ay_end])
                    except Exception:
                        pass

                where_sql = (" WHERE " + " AND ".join(where)) if where else ""
                with conn.cursor() as cur:
                    # Total
                    cur.execute(f"SELECT COUNT(*) AS total FROM violations v{where_sql}", params)
                    total = (cur.fetchone() or {}).get('total', 0)

                    # By college
                    cur.execute(
                        f"SELECT COALESCE(s.college,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC"
                        , params)
                    by_college = cur.fetchall() or []

                    # By program (course)
                    cur.execute(
                        f"SELECT COALESCE(s.course,'Unknown') AS label, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY label ORDER BY cnt DESC"
                        , params)
                    by_program = cur.fetchall() or []

                    # By gender (male/female only)
                    cur.execute(
                        f"SELECT LOWER(COALESCE(s.gender,'')) AS g, COUNT(*) AS cnt FROM violations v LEFT JOIN students s ON v.student_id=s.student_id{where_sql} GROUP BY g"
                        , params)
                    raw_gender = cur.fetchall() or []
                    counts = {str((row.get('g') or '')).strip().lower(): int(row.get('cnt') or 0) for row in raw_gender}
                    by_gender = [
                        {'label': 'male', 'cnt': counts.get('male', 0)},
                        {'label': 'female', 'cnt': counts.get('female', 0)},
                    ]

                    # Resolved vs unresolved
                    cur.execute(
                        f"SELECT v.status AS label, COUNT(*) AS cnt FROM violations v{where_sql} GROUP BY v.status"
                        , params)
                    by_status = cur.fetchall() or []

                    return {
                        "total": total,
                        "by_college": by_college,
                        "by_program": by_program,
                        "by_gender": by_gender,
                        "by_status": by_status,
                    }
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB analytics error: {e}")
            return {"total": 0, "by_college": [], "by_program": [], "by_gender": [], "by_status": []}

    def update_violation_status(self, violation_id: int, status: str) -> bool:
        """Update violation status. Allowed: pending, forwarded, resolved."""
        allowed = {"pending", "forwarded", "resolved"}
        if status not in allowed:
            return False
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE violations SET status=%s WHERE violation_id=%s",
                        (status, violation_id)
                    )
                    return True
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB update_violation_status error: {e}")
            return False

    def get_violations_trend(self, start_dt: Optional[str] = None, end_dt: Optional[str] = None,
                              academic_year: Optional[str] = None, semester: Optional[str] = None,
                              group_by: str = 'day') -> Dict[str, Any]:
        """Return time series counts grouped by day/week/month."""
        try:
            conn = self.get_connection()
            try:
                where = []
                params = []
                if start_dt:
                    where.append("timestamp >= %s")
                    params.append(start_dt)
                if end_dt:
                    where.append("timestamp <= %s")
                    params.append(end_dt)
                if academic_year and semester in {"1", "2"}:
                    try:
                        start_year = int(academic_year.split('-')[0])
                        if semester == "1":
                            ay_start = f"{start_year}-08-01 00:00:00"
                            ay_end = f"{start_year}-12-31 23:59:59"
                        else:
                            ay_start = f"{start_year+1}-01-01 00:00:00"
                            ay_end = f"{start_year+1}-05-31 23:59:59"
                        where.append("timestamp BETWEEN %s AND %s")
                        params.extend([ay_start, ay_end])
                    except Exception:
                        pass

                where_sql = (" WHERE " + " AND ".join(where)) if where else ""

                if group_by == 'month':
                    group_expr = "DATE_FORMAT(timestamp, '%Y-%m')"
                    order_expr = "DATE_FORMAT(timestamp, '%Y-%m')"
                elif group_by == 'week':
                    group_expr = "YEARWEEK(timestamp, 3)"  # ISO week
                    order_expr = "YEARWEEK(timestamp, 3)"
                else:
                    group_expr = "DATE(timestamp)"
                    order_expr = "DATE(timestamp)"

                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {group_expr} AS label, COUNT(*) AS cnt FROM violations{where_sql} GROUP BY label ORDER BY {order_expr} ASC",
                        params
                    )
                    rows = cur.fetchall() or []
                    return {"series": rows}
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB trend error: {e}")
            return {"series": []}


# Global database manager instance
db_manager = DatabaseManager()
