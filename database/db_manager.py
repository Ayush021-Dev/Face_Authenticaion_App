# face_auth_system/database/db_manager.py
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from datetime import datetime
import pickle
import numpy as np

class DatabaseManager:
    def __init__(self):
        try:
            print("Debug - Attempting to connect to MySQL database...")
            print("Debug - Connection parameters:")
            print("  Host: localhost")
            print("  User: faceuser")
            print("  Database: face_auth_new")
            
            self.connection = mysql.connector.connect(
                host='localhost',
                user='faceuser',
                password='1234',
                database='face_auth_new'
            )
            
            if self.connection.is_connected():
                db_info = self.connection.get_server_info()
                print(f"Debug - Connected to MySQL Server version {db_info}")
                
                self.cursor = self.connection.cursor(buffered=True)
                print("Debug - Database cursor created")
                
                # Test the connection with a simple query
                self.cursor.execute("SELECT DATABASE();")
                db_name = self.cursor.fetchone()
                print(f"Debug - Connected to database: {db_name[0]}")
                
                self.create_tables()
            else:
                print("Debug - Failed to connect to database")
        except Error as err:
            print(f"Error connecting to database: {err}")
            print(f"Error code: {err.errno}")
            print(f"Error message: {err.msg}")
            raise err

    def create_tables(self):
        try:
            # Drop existing tables to ensure clean migration
            self.cursor.execute("DROP TABLE IF EXISTS login_logs")
            self.cursor.execute("DROP TABLE IF EXISTS employees")
            self.cursor.execute("DROP TABLE IF EXISTS equipment_areas")
            
            # Equipment areas table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS equipment_areas (
                equipment_id VARCHAR(20) PRIMARY KEY,
                equipment_name VARCHAR(100) NOT NULL,
                area_lat FLOAT NOT NULL,
                area_lon FLOAT NOT NULL,
                area_radius FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Employees table with equipment area reference
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                emp_id VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                face_encoding LONGBLOB NOT NULL,
                equipment_id VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (equipment_id) REFERENCES equipment_areas(equipment_id)
            )
            ''')
            
            # Login logs table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_logs (
                log_id INT AUTO_INCREMENT PRIMARY KEY,
                emp_id VARCHAR(20),
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                latitude FLOAT,
                longitude FLOAT,
                FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
            )
            ''')
            
            self.connection.commit()
            print("Tables created successfully")
        except Error as err:
            print(f"Error creating tables: {err}")
            raise err

    def close_connection(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'connection') and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed")

    def register_employee(self, emp_id, name, face_encoding, equipment_id):
        try:
            # First check if employee already exists
            check_query = "SELECT emp_id FROM employees WHERE emp_id = %s"
            self.cursor.execute(check_query, (emp_id,))
            if self.cursor.fetchone():
                print(f"Debug - Employee {emp_id} already exists in database")
                return False
            
            # Check if equipment area exists
            if not self.get_equipment_area(equipment_id):
                print(f"Debug - Equipment area {equipment_id} does not exist")
                return False
            
            query = "INSERT INTO employees (emp_id, name, face_encoding, equipment_id) VALUES (%s, %s, %s, %s)"
            self.cursor.execute(query, (emp_id, name, face_encoding, equipment_id))
            self.connection.commit()
            return True
        except mysql.connector.Error as err:
            print(f"Error registering employee: {err}")
            return False

    def update_employee_area(self, emp_id, area_lat, area_lon, area_radius):
        try:
            print(f"Debug - Starting area update process")
            print(f"Debug - Employee ID: {emp_id}")
            print(f"Debug - Area values:")
            print(f"  Latitude: {area_lat} (type: {type(area_lat)})")
            print(f"  Longitude: {area_lon} (type: {type(area_lon)})")
            print(f"  Radius: {area_radius} (type: {type(area_radius)})")
            
            # First verify the employee exists
            check_query = "SELECT emp_id, name FROM employees WHERE emp_id = %s"
            self.cursor.execute(check_query, (emp_id,))
            result = self.cursor.fetchone()
            if not result:
                print(f"Debug - Employee {emp_id} not found in database")
                return False
            print(f"Debug - Found employee: {result}")
            
            # Now try the update
            query = "UPDATE employees SET area_lat = %s, area_lon = %s, area_radius = %s WHERE emp_id = %s"
            print(f"Debug - Query: {query}")
            print(f"Debug - Parameters: {area_lat}, {area_lon}, {area_radius}, {emp_id}")
            
            self.cursor.execute(query, (area_lat, area_lon, area_radius, emp_id))
            rows_affected = self.cursor.rowcount
            print(f"Debug - Rows affected: {rows_affected}")
            
            self.connection.commit()
            print("Debug - Update successful")
            
            # Verify the update
            verify_query = "SELECT emp_id, name, area_lat, area_lon, area_radius FROM employees WHERE emp_id = %s"
            self.cursor.execute(verify_query, (emp_id,))
            result = self.cursor.fetchone()
            print(f"Debug - Verification after update: {result}")
            
            return True
        except mysql.connector.Error as err:
            print(f"Error updating employee area: {err}")
            print(f"Error code: {err.errno}")
            print(f"Error message: {err.msg}")
            return False
    
    def log_login(self, emp_id, latitude, longitude):
        try:
            query = "INSERT INTO login_logs (emp_id, latitude, longitude) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (emp_id, latitude, longitude))
            self.connection.commit()
            return True
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return False
    
    def get_employee(self, emp_id):
        query = """
        SELECT e.emp_id, e.name, e.face_encoding, e.equipment_id,
               ea.area_lat, ea.area_lon, ea.area_radius
        FROM employees e
        LEFT JOIN equipment_areas ea ON e.equipment_id = ea.equipment_id
        WHERE e.emp_id = %s
        """
        self.cursor.execute(query, (emp_id,))
        return self.cursor.fetchone()
    
    def get_all_employees(self):
        try:
            query = "SELECT emp_id, name, face_encoding FROM employees"
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return []
    
    def get_login_history(self, emp_id=None, limit=10):
        try:
            if emp_id:
                query = """
                SELECT l.log_id, e.name, l.login_time, l.latitude, l.longitude 
                FROM login_logs l
                JOIN employees e ON l.emp_id = e.emp_id
                WHERE l.emp_id = %s
                ORDER BY l.login_time DESC
                LIMIT %s
                """
                self.cursor.execute(query, (emp_id, limit))
            else:
                query = """
                SELECT l.log_id, e.name, l.login_time, l.latitude, l.longitude 
                FROM login_logs l
                JOIN employees e ON l.emp_id = e.emp_id
                ORDER BY l.login_time DESC
                LIMIT %s
                """
                self.cursor.execute(query, (limit,))
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return []
    
    def face_exists(self, face_encoding, similarity_threshold=0.6):
        """
        Check if a face encoding already exists in the database
        
        Args:
            face_encoding: The face encoding to check
            similarity_threshold: Threshold for determining if faces match
            
        Returns:
            (exists, emp_id, name) or (False, None, None) if no match
        """
        try:
            query = "SELECT emp_id, name, face_encoding FROM employees"
            self.cursor.execute(query)
            employees = self.cursor.fetchall()
            
            for emp_id, name, db_encoding in employees:
                # Decode stored encoding
                stored_encoding = pickle.loads(db_encoding)
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(stored_encoding - face_encoding)
                
                # If distance is below threshold, faces match
                if distance < similarity_threshold:
                    return True, emp_id, name
            
            return False, None, None
        except mysql.connector.Error as err:
            print(f"Error checking face existence: {err}")
            return False, None, None

    def add_equipment_area(self, equipment_id, equipment_name, area_lat, area_lon, area_radius):
        try:
            query = """
            INSERT INTO equipment_areas (equipment_id, equipment_name, area_lat, area_lon, area_radius)
            VALUES (%s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (equipment_id, equipment_name, area_lat, area_lon, area_radius))
            self.connection.commit()
            return True
        except mysql.connector.Error as err:
            print(f"Error adding equipment area: {err}")
            return False

    def get_equipment_area(self, equipment_id):
        try:
            query = "SELECT * FROM equipment_areas WHERE equipment_id = %s"
            self.cursor.execute(query, (equipment_id,))
            return self.cursor.fetchone()
        except mysql.connector.Error as err:
            print(f"Error getting equipment area: {err}")
            return None

    def get_all_equipment_areas(self):
        try:
            query = "SELECT * FROM equipment_areas"
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"Error getting equipment areas: {err}")
            return []