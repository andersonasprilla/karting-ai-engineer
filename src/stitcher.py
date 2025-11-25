import pandas as pd
import numpy as np
import os

def load_and_stitch_from_folder(folder_path):
    """
    Reads distinct CSV files from a folder (RPM, GPS, Steering) 
    and merges them into one master timeline.
    """
    print(f"🔧 Looking for files in: {folder_path}")
    
    # 1. Define the specific filenames we expect from RaceStudio
    # You can add more here if you export more channels later
    files = {
        'rpm': 'RPM.csv',
        'gps': '_GPS.csv',
        'steer': 'Steering Angle.csv',
        'gyro': 'GyroZ.csv'
    }
    
    data_frames = {}

    # 2. Load each file safely
    for key, filename in files.items():
        path = os.path.join(folder_path, filename)
        if not os.path.exists(path):
            print(f"⚠️ Warning: {filename} not found. Skipping.")
            continue
            
        try:
            # RaceStudio often puts units in the 2nd row (row index 1). We skip it.
            # GPS file usually doesn't have units in 2nd row, so we check.
            if key == 'gps':
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, skiprows=[1])
                
            # Standardize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Ensure time is numeric and sorted
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time')
            
            data_frames[key] = df
            print(f"✅ Loaded {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    if 'rpm' not in data_frames:
        return None, "RPM file is missing. This is required for the timeline."

    # 3. Start Merging
    # We use RPM as the "Master Clock" because it has the highest sample rate (50Hz)
    master = data_frames['rpm'].rename(columns={'value': 'rpm'})[['time', 'rpm']]

    # Merge GPS (slower 10Hz) - 'backward' direction fills gaps with the last known position
    if 'gps' in data_frames:
        gps_cols = ['time', 'speed', 'lat', 'lon']
        gps_data = data_frames['gps'][gps_cols]
        master = pd.merge_asof(master, gps_data, on='time', direction='nearest')

    # Merge Steering
    if 'steer' in data_frames:
        steer_data = data_frames['steer'].rename(columns={'value': 'steer'})[['time', 'steer']]
        master = pd.merge_asof(master, steer_data, on='time', direction='nearest')

    # Merge Gyro (Yaw Rate)
    if 'gyro' in data_frames:
        gyro_data = data_frames['gyro'].rename(columns={'value': 'yaw_rate'})[['time', 'yaw_rate']]
        master = pd.merge_asof(master, gyro_data, on='time', direction='nearest')
    
    # 4. Convert Speed to MPH
    if 'speed' in master.columns:
        # Assuming GPS speed is m/s (standard). 1 m/s = 2.237 mph
        master['speed_mph'] = master['speed'] * 2.237
        
        # 4b. Calculate Longitudinal G (Braking/Acceleration)
        # diff() calculates the change between this row and the previous one
        # 50Hz data = 0.02s per step
        master['delta_speed_ms'] = master['speed'].diff()
        master['delta_time'] = master['time'].diff()
        
        # Accel (m/s^2) = Delta Speed / Delta Time
        # G-Force = Accel / 9.81
        master['long_g'] = (master['delta_speed_ms'] / master['delta_time']) / 9.81
        
        # Smooth it out (Rolling average) to remove GPS jitters
        master['long_g'] = master['long_g'].rolling(window=10).mean()

        # 4c. Calculate Lateral G (Cornering Force)
        if 'yaw_rate' in master.columns:
            # Convert deg/s to rad/s
            master['yaw_rad'] = np.radians(master['yaw_rate'])
            # Centripetal Force: a = v * omega
            master['lat_g'] = (master['speed'] * master['yaw_rad']) / 9.81
            # Invert if necessary (depending on if Left is positive or negative)
            master['lat_g'] = master['lat_g'] * -1 
    
    # Clean up NaNs created by the diff/rolling functions
    master = master.fillna(0)
    
    # Select only what the AI needs to see
    final_columns = ['time', 'rpm', 'speed_mph', 'steer', 'long_g', 'lat_g']
    # Keep only columns that actually exist
    final_columns = [c for c in final_columns if c in master.columns]
    
    return master[final_columns], "Success"

# Simple test block to run this script directly
if __name__ == "__main__":
    # Create a 'test_data' folder and put your CSVs there to test this
    test_dir = "data" 
    if os.path.exists(test_dir):
        df, msg = load_and_stitch_from_folder(test_dir)
        if df is not None:
            print(df.head())
            df.to_csv("master_test.csv", index=False)
            print("💾 Saved master_test.csv")
    else:
        print("Please create a 'data' folder and put your CSVs in it to test.")