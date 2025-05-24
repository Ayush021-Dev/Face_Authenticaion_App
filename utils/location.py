# face_auth_system/utils/location.py
import requests
from math import radians, sin, cos, sqrt, atan2

def get_location():
    """
    Get current geolocation using IP address
    
    Returns:
        Tuple of (latitude, longitude) or (None, None) if error
    """
    try:
        # Using ip-api for geolocation
        response = requests.get('http://ip-api.com/json/')
        data = response.json()
        
        if data['status'] == 'success':
            return data['lat'], data['lon']
        return None, None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None

def is_within_area(current_lat, current_lon, area_lat, area_lon, area_radius):
    """
    Check if current location is within the defined area
    
    Args:
        current_lat: Current latitude
        current_lon: Current longitude
        area_lat: Center latitude of defined area
        area_lon: Center longitude of defined area
        area_radius: Radius in kilometers
        
    Returns:
        Boolean indicating if location is within area
    """
    if current_lat is None or current_lon is None:
        return False
        
    if area_lat is None or area_lon is None or area_radius is None:
        return True  # If no area defined, accept any location
    
    # Calculate distance using Haversine formula
    R = 6371  # Earth radius in kilometers
    
    # Convert coordinates to radians
    lat1, lon1 = radians(current_lat), radians(current_lon)
    lat2, lon2 = radians(area_lat), radians(area_lon)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance <= area_radius