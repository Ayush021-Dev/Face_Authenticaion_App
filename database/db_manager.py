import sqlite3
import pickle
import numpy as np
from datetime import datetime
import os
from typing import List, Tuple, Optional, Dict, Any

class DatabaseManager:
    def __init__(self, db_path: str = "face_auth_system.db"):
        """Initialize SQLite database manager"""
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Create admins table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admins (
                    admin_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create equipment_areas table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equipment_areas (
                    area_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_name TEXT UNIQUE NOT NULL,
                    area_lat REAL NOT NULL,
                    area_lon REAL NOT NULL,
                    area_radius REAL NOT NULL,
                    num_equipment INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create employees table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    emp_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    face_encoding BLOB NOT NULL,
                    equipment_area TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (equipment_area) REFERENCES equipment_areas (equipment_name)
                )
            ''')
            
            # Create equipment table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equipment (
                    equipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_name TEXT NOT NULL,
                    area_name TEXT NOT NULL,
                    equipment_type TEXT DEFAULT 'General',
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (area_name) REFERENCES equipment_areas (equipment_name)
                )
            ''')
            
            # Create parameters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameters (
                    parameter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_id INTEGER NOT NULL,
                    parameter_name TEXT NOT NULL,
                    unit TEXT DEFAULT '',
                    min_value REAL,
                    max_value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (equipment_id) REFERENCES equipment (equipment_id),
                    UNIQUE(equipment_id, parameter_name)
                )
            ''')
            
            # Create equipment_readings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equipment_readings (
                    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_id INTEGER NOT NULL,
                    parameter_id INTEGER NOT NULL,
                    value REAL NOT NULL,
                    reading_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    recorded_by TEXT,
                    FOREIGN KEY (equipment_id) REFERENCES equipment (equipment_id),
                    FOREIGN KEY (parameter_id) REFERENCES parameters (parameter_id),
                    FOREIGN KEY (recorded_by) REFERENCES employees (emp_id)
                )
            ''')
            
            # Create login_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS login_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emp_id TEXT NOT NULL,
                    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    latitude REAL,
                    longitude REAL,
                    FOREIGN KEY (emp_id) REFERENCES employees (emp_id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_employees_area ON employees (equipment_area)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_equipment_area ON equipment (area_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_readings_equipment ON equipment_readings (equipment_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_readings_time ON equipment_readings (reading_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_login_logs_emp ON login_logs (emp_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_login_logs_time ON login_logs (login_time)')
            
            conn.commit()
            print("Database initialized successfully")
            
        except Exception as e:
            conn.rollback()
            print(f"Error initializing database: {e}")
            raise
        finally:
            conn.close()
    
    # Admin Management
    def add_admin(self, admin_id: str, name: str, password: str) -> bool:
        """Add new admin"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO admins (admin_id, name, password) VALUES (?, ?, ?)",
                (admin_id, name, password)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Admin already exists
        except Exception as e:
            print(f"Error adding admin: {e}")
            return False
        finally:
            conn.close()
    
    def verify_admin(self, admin_id: str, password: str) -> bool:
        """Verify admin credentials"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT password FROM admins WHERE admin_id = ?",
                (admin_id,)
            )
            result = cursor.fetchone()
            return result and result[0] == password
        except Exception as e:
            print(f"Error verifying admin: {e}")
            return False
        finally:
            conn.close()
    
    # Equipment Areas Management
    def add_equipment_area(self, equipment_name: str, area_lat: float, area_lon: float, 
                          area_radius: float, num_equipment: int = 1) -> bool:
        """Add new equipment area"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO equipment_areas (equipment_name, area_lat, area_lon, area_radius, num_equipment) VALUES (?, ?, ?, ?, ?)",
                (equipment_name, area_lat, area_lon, area_radius, num_equipment)
            )
            conn.commit()
            
            # Auto-create equipment entries for this area
            self._create_default_equipment(equipment_name, num_equipment)
            
            return True
        except sqlite3.IntegrityError:
            return False  # Area already exists
        except Exception as e:
            print(f"Error adding equipment area: {e}")
            return False
        finally:
            conn.close()
    
    def _create_default_equipment(self, area_name: str, num_equipment: int):
        """Create default equipment entries for an area"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for i in range(1, num_equipment + 1):
                equipment_name = f"{area_name} Unit {i}" if num_equipment > 1 else area_name
                cursor.execute(
                    "INSERT INTO equipment (equipment_name, area_name, equipment_type) VALUES (?, ?, ?)",
                    (equipment_name, area_name, "Process Unit")
                )
                
                equipment_id = cursor.lastrowid
                
                # Add default parameters
                default_params = [
                    ("Temperature", "°C", 0, 200),
                    ("Pressure", "bar", 0, 50),
                    ("Flow Rate", "m³/h", 0, 1000),
                    ("Level", "%", 0, 100)
                ]
                
                for param_name, unit, min_val, max_val in default_params:
                    cursor.execute(
                        "INSERT INTO parameters (equipment_id, parameter_name, unit, min_value, max_value) VALUES (?, ?, ?, ?, ?)",
                        (equipment_id, param_name, unit, min_val, max_val)
                    )
            
            conn.commit()
        except Exception as e:
            print(f"Error creating default equipment: {e}")
        finally:
            conn.close()
    
    def get_all_equipment_areas(self) -> List[Dict]:
        """Get all equipment areas"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT area_id, equipment_name, area_lat, area_lon, area_radius, num_equipment FROM equipment_areas ORDER BY equipment_name"
            )
            results = cursor.fetchall()
            
            # Convert to list of dictionaries for easier access
            areas = []
            for row in results:
                areas.append({
                    'area_id': row[0],
                    'equipment_name': row[1],
                    'area_lat': row[2],
                    'area_lon': row[3], 
                    'area_radius': row[4],
                    'num_equipment': row[5]
                })
            
            return areas
        except Exception as e:
            print(f"Error getting equipment areas: {e}")
            return []
        finally:
            conn.close()
    
    # Employee Management
    def register_employee(self, emp_id: str, name: str, face_encoding: bytes, equipment_area: str = None) -> bool:
        """Register new employee"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO employees (emp_id, name, face_encoding, equipment_area) VALUES (?, ?, ?, ?)",
                (emp_id, name, face_encoding, equipment_area)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Employee already exists
        except Exception as e:
            print(f"Error registering employee: {e}")
            return False
        finally:
            conn.close()
    
    def get_employee(self, emp_id: str) -> Optional[Dict]:
        """Get employee details with area information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT e.emp_id, e.name, e.face_encoding, e.equipment_area,
                       ea.area_lat, ea.area_lon, ea.area_radius
                FROM employees e
                LEFT JOIN equipment_areas ea ON e.equipment_area = ea.equipment_name
                WHERE e.emp_id = ?
            ''', (emp_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'emp_id': result[0],
                    'name': result[1],
                    'face_encoding': result[2],
                    'equipment_name': result[3],
                    'area_lat': result[4],
                    'area_lon': result[5],
                    'area_radius': result[6]
                }
            return None
        except Exception as e:
            print(f"Error getting employee: {e}")
            return None
        finally:
            conn.close()
    
    def get_all_employees(self) -> List[Tuple]:
        """Get all employees"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT emp_id, name, face_encoding, equipment_area FROM employees")
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting employees: {e}")
            return []
        finally:
            conn.close()
    
    def update_employee_equipment(self, emp_id: str, equipment_area: str = None) -> bool:
        """Update employee equipment assignment"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "UPDATE employees SET equipment_area = ? WHERE emp_id = ?",
                (equipment_area, emp_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating employee equipment: {e}")
            return False
        finally:
            conn.close()
    
    def face_exists(self, face_encoding: np.ndarray, threshold: float = 0.6) -> Tuple[bool, str, str]:
        """Check if face already exists in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT emp_id, name, face_encoding FROM employees")
            employees = cursor.fetchall()
            
            for emp_id, name, stored_encoding_blob in employees:
                stored_encoding = pickle.loads(stored_encoding_blob)
                distance = np.linalg.norm(face_encoding - stored_encoding)
                
                if distance < threshold:
                    return True, emp_id, name
            
            return False, "", ""
        except Exception as e:
            print(f"Error checking face existence: {e}")
            return False, "", ""
        finally:
            conn.close()
    
    # Equipment Management
    def get_equipment_by_area(self, area_name: str) -> List[Tuple]:
        """Get all equipment for a specific area with parameters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT e.equipment_id, e.equipment_name, e.area_name, e.equipment_type,
                       GROUP_CONCAT(p.parameter_name, ', ') as parameters
                FROM equipment e
                LEFT JOIN parameters p ON e.equipment_id = p.equipment_id
                WHERE e.area_name = ?
                GROUP BY e.equipment_id, e.equipment_name, e.area_name, e.equipment_type
                ORDER BY e.equipment_name
            ''', (area_name,))
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting equipment by area: {e}")
            return []
        finally:
            conn.close()
    
    def get_equipment_parameters(self, equipment_id: int) -> List[Tuple]:
        """Get all parameters for specific equipment"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT parameter_id, parameter_name, unit, min_value, max_value FROM parameters WHERE equipment_id = ? ORDER BY parameter_name",
                (equipment_id,)
            )
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting equipment parameters: {e}")
            return []
        finally:
            conn.close()
    
    # Readings Management
    def add_equipment_reading(self, equipment_id: int, parameter_id: int, value: float, recorded_by: str = None) -> bool:
        """Add equipment reading"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO equipment_readings (equipment_id, parameter_id, value, recorded_by) VALUES (?, ?, ?, ?)",
                (equipment_id, parameter_id, value, recorded_by)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding equipment reading: {e}")
            return False
        finally:
            conn.close()
    
    def get_equipment_readings(self, equipment_id: int, limit: int = 100) -> List[Tuple]:
        """Get readings for specific equipment"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT er.reading_id, er.equipment_id, er.parameter_id, er.value, 
                       er.reading_time, p.parameter_name, p.unit
                FROM equipment_readings er
                JOIN parameters p ON er.parameter_id = p.parameter_id
                WHERE er.equipment_id = ?
                ORDER BY er.reading_time DESC
                LIMIT ?
            ''', (equipment_id, limit))
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting equipment readings: {e}")
            return []
        finally:
            conn.close()
    
    def get_all_readings_for_area(self, area_name: str, limit: int = 1000) -> List[Tuple]:
        """Get all readings for equipment in a specific area"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT er.reading_id, er.equipment_id, er.parameter_id, er.value, 
                       er.reading_time, p.parameter_name, p.unit, e.equipment_name
                FROM equipment_readings er
                JOIN parameters p ON er.parameter_id = p.parameter_id
                JOIN equipment e ON er.equipment_id = e.equipment_id
                WHERE e.area_name = ?
                ORDER BY er.reading_time DESC
                LIMIT ?
            ''', (area_name, limit))
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting area readings: {e}")
            return []
        finally:
            conn.close()
    
    # Login Logging
    def log_login(self, emp_id: str, latitude: float = None, longitude: float = None) -> bool:
        """Log employee login"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO login_logs (emp_id, latitude, longitude) VALUES (?, ?, ?)",
                (emp_id, latitude, longitude)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error logging login: {e}")
            return False
        finally:
            conn.close()
    
    def get_login_history(self, emp_id: str = None, limit: int = 50) -> List[Tuple]:
        """Get login history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if emp_id:
                cursor.execute('''
                    SELECT ll.log_id, ll.emp_id, e.name, ll.login_time, ll.latitude, ll.longitude
                    FROM login_logs ll
                    JOIN employees e ON ll.emp_id = e.emp_id
                    WHERE ll.emp_id = ?
                    ORDER BY ll.login_time DESC
                    LIMIT ?
                ''', (emp_id, limit))
            else:
                cursor.execute('''
                    SELECT ll.log_id, ll.emp_id, e.name, ll.login_time, ll.latitude, ll.longitude
                    FROM login_logs ll
                    JOIN employees e ON ll.emp_id = e.emp_id
                    ORDER BY ll.login_time DESC
                    LIMIT ?
                ''', (limit,))
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting login history: {e}")
            return []
        finally:
            conn.close()
    
    # Utility Methods
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Count records in each table
            tables = ['admins', 'equipment_areas', 'employees', 'equipment', 'parameters', 'equipment_readings', 'login_logs']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            return stats
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
        finally:
            conn.close()
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def close_connection(self):
        """Close database connection (for cleanup)"""
        # SQLite connections are closed automatically when the connection object is destroyed
        # This method is here for compatibility with the original interface
        pass