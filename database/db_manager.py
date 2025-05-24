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
            self.connection = mysql.connector.connect(
                host='localhost',
                user='faceuser',
                password='1234',
                database='face_auth_new'
            )
            if self.connection.is_connected():
                print("Successfully connected to MySQL database")
                self.cursor = self.connection.cursor(buffered=True)
                self.create_tables()
        except Error as err:
            print(f"Error: {err}")
            raise err

    def create_tables(self):
        try:
            # Employees table with area location fields
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                emp_id VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                face_encoding LONGBLOB NOT NULL,
                area_lat FLOAT,
                area_lon FLOAT, 
                area_radius FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    def register_employee(self, emp_id, name, face_encoding):
        try:
            query = "INSERT INTO employees (emp_id, name, face_encoding) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (emp_id, name, face_encoding))
            self.connection.commit()
            return True
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return False
    
    def update_employee_area(self, emp_id, area_lat, area_lon, area_radius):
        try:
            query = "UPDATE employees SET area_lat = %s, area_lon = %s, area_radius = %s WHERE emp_id = %s"
            self.cursor.execute(query, (area_lat, area_lon, area_radius, emp_id))
            self.connection.commit()
            return True
        except mysql.connector.Error as err:
            print(f"Error: {err}")
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
        query = "SELECT emp_id, name, face_encoding, area_lat, area_lon, area_radius FROM employees WHERE emp_id = %s"
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