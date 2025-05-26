# face_auth_system/utils/location.py
import requests
from math import radians, sin, cos, sqrt, atan2
import time

def get_location():
    """
    Get current geolocation using IP address
    
    Returns:
        Tuple of (latitude, longitude) or (None, None) if error
    """
    try:
        # Using ip-api for geolocation
        response = requests.get('http://ip-api.com/json/', timeout=5)
        data = response.json()
        
        if data['status'] == 'success':
            return data['lat'], data['lon']
        return None, None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None

def get_precise_location():
    """
    Attempt to get more precise location using multiple services
    
    Returns:
        Tuple of (latitude, longitude) or (None, None) if error
    """
    services = [
        'http://ip-api.com/json/',
        'https://ipapi.co/json/',
        'https://ipinfo.io/json'
    ]
    
    for service in services:
        try:
            response = requests.get(service, timeout=3)
            data = response.json()
            
            # Handle different response formats
            if service == 'http://ip-api.com/json/':
                if data.get('status') == 'success':
                    return data['lat'], data['lon']
            elif service == 'https://ipapi.co/json/':
                if 'latitude' in data and 'longitude' in data:
                    return data['latitude'], data['longitude']
            elif service == 'https://ipinfo.io/json':
                if 'loc' in data:
                    lat, lon = data['loc'].split(',')
                    return float(lat), float(lon)
                    
        except Exception as e:
            print(f"Error with service {service}: {e}")
            continue
    
    return None, None

def is_within_area(current_lat, current_lon, area_lat, area_lon, area_radius):
    """
    Check if current location is within the defined equipment area
    
    Args:
        current_lat: Current latitude
        current_lon: Current longitude
        area_lat: Center latitude of equipment area
        area_lon: Center longitude of equipment area
        area_radius: Radius in kilometers
        
    Returns:
        Tuple of (is_within, distance) - Boolean and distance in km
    """
    if current_lat is None or current_lon is None:
        return False, None
        
    if area_lat is None or area_lon is None or area_radius is None:
        return False, None
    
    # Calculate distance using Haversine formula
    distance = calculate_distance(current_lat, current_lon, area_lat, area_lon)
    
    return distance <= area_radius, distance

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert coordinates to radians
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def check_multiple_areas(current_lat, current_lon, areas_list):
    """
    Check which equipment areas the current location falls within
    
    Args:
        current_lat: Current latitude
        current_lon: Current longitude
        areas_list: List of tuples (area_id, area_name, area_lat, area_lon, area_radius)
        
    Returns:
        List of dictionaries with area info and distances for areas within range
    """
    valid_areas = []
    
    if current_lat is None or current_lon is None:
        return valid_areas
    
    for area in areas_list:
        area_id, area_name, area_lat, area_lon, area_radius = area[:5]
        
        is_within, distance = is_within_area(
            current_lat, current_lon, 
            area_lat, area_lon, area_radius
        )
        
        if is_within:
            valid_areas.append({
                'area_id': area_id,
                'area_name': area_name,
                'distance': distance,
                'area_lat': area_lat,
                'area_lon': area_lon,
                'area_radius': area_radius
            })
    
    # Sort by distance (closest first)
    valid_areas.sort(key=lambda x: x['distance'])
    return valid_areas

def get_location_with_retry(max_retries=3):
    """
    Get location with retry mechanism
    
    Args:
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if all attempts fail
    """
    for attempt in range(max_retries):
        lat, lon = get_location()
        if lat is not None and lon is not None:
            return lat, lon
        
        if attempt < max_retries - 1:
            print(f"Location attempt {attempt + 1} failed, retrying...")
            time.sleep(1)
    
    print("Failed to get location after all retries")
    return None, None

def format_coordinates(lat, lon, precision=6):
    """
    Format coordinates for display
    
    Args:
        lat: Latitude
        lon: Longitude
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if lat is None or lon is None:
        return "Location unavailable"
    
    return f"{lat:.{precision}f}, {lon:.{precision}f}"

def get_location_info(lat, lon):
    """
    Get additional location information (city, country, etc.)
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dictionary with location details or None if error
    """
    try:
        response = requests.get(
            f'http://ip-api.com/json/?lat={lat}&lon={lon}',
            timeout=5
        )
        data = response.json()
        
        if data.get('status') == 'success':
            return {
                'city': data.get('city', 'Unknown'),
                'region': data.get('regionName', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'timezone': data.get('timezone', 'Unknown')
            }
    except Exception as e:
        print(f"Error getting location info: {e}")
    
    return None