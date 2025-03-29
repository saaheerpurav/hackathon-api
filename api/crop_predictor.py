import pandas as pd
import math

import numpy as np
import pickle
import requests

import warnings
warnings.filterwarnings('ignore')

with open('LogReg.pkl', 'rb') as file:
    model = pickle.load(file)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth (specified in decimal degrees)
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    return c * r

def find_nearest_district(user_lat, user_lon):
    # Read district data
    try:
        df = pd.read_csv("merged_districts_data.csv")
    except FileNotFoundError:
        print("Error: merged_districts_data.csv not found!")
        return

    # Calculate distances
    distances = []
    for _, row in df.iterrows():
        dist = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
        distances.append(dist)
    
    # Find minimum distance
    min_index = distances.index(min(distances))
    return df.iloc[min_index]

def get_rainfall(lat, lon, year=2024):
    """
    Fetch annual rainfall (in mm) for given coordinates and year.
    Returns None on failure.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": "precipitation_sum"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise HTTP errors

        data = response.json()
        rainfall_data = data.get("daily", {}).get("precipitation_sum", [])

        if not rainfall_data:
            print("No rainfall data found in response")
            return None

        return sum(rainfall_data)

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
    except ValueError as e:
        print(f"Invalid JSON response: {str(e)}")

    return None

def get_weather(user_lat, user_lon):
    response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?lat={user_lat}&lon={user_lon}&units=metric&appid=0acbed08482b338a220e6ba9c72d00e9")
    data = response.json()

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": get_rainfall(user_lat, user_lon)
    }

def get_predicted_crop(user_lat, user_lon):    
    nearest = find_nearest_district(user_lat, user_lon)
    weather = get_weather(user_lat, user_lon)

    #N,P,K,temperature,humidity,ph,rainfall
    data = np.array([[nearest['N'], nearest['P'], nearest['K'], weather["temp"], weather["humidity"], nearest['pH'], weather["rainfall"]]])
    prediction = model.predict(data)
    
    return prediction[0]