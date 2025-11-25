import pandas as pd
import numpy as np
import os
import glob

def load_and_stitch_from_folder(folder_path):
    print(f"🔧 Scanning files in: {folder_path}")
    all_csvs = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # STRATEGY 1: Check for a "Single File Export" (like 4.csv)
    # This file type contains everything in one sheet but has metadata at the top.
    for f in all_csvs:
        try:
            # Peek at the file to see if it has the tell-tale header
            with open(f, 'r') as open_f:
                lines = [open_f.readline() for _ in range(20)]
            
            # Look for the header line index
            header_idx = -1
            for i, line in enumerate(lines):
                if '"Time"' in line and '"GPS Speed"' in line:
                    header_idx = i
                    break
            
            if header_idx != -1:
                print(f"✅ Detected Single-File Export: {os.path.basename(f)}")
                return process_single_file(f, header_idx)
        except Exception as e:
            continue

    # STRATEGY 2: Fallback to "Batch Stitching" (RPM.csv + _GPS.csv, etc.)
    return process_batch_files(folder_path)

def process_single_file(file_path, header_idx):
    """Handles the all-in-one CSV format."""
    try:
        # 1. Parse Metadata for Beacon Markers (Laps)
        beacon_markers = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('"Beacon Markers"'):
                    parts = line.strip().split(',')
                    # Extract numbers, removing quotes
                    for p in parts[1:]:
                        clean_p = p.replace('"', '')
                        try:
                            beacon_markers.append(float(clean_p))
                        except:
                            pass
                    break
        
        # 2. Load Data (Using the detected header row)
        # Note: We subtract 1 from the index because read_csv counts differently than readlines
        df = pd.read_csv(file_path, header=header_idx-1, on_bad_lines='skip')
        
        # Drop the "Units" row (always the first row after header)
        df = df.drop(0).reset_index(drop=True)
        
        # Clean Columns
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        
        # Convert Types
        numeric_cols = ['Time', 'RPM', 'GPS Speed', 'Steering Angle', 'GPS LatAcc', 'GPS LonAcc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Assign Laps based on Beacon Markers
        df['lap'] = 1 # Default to Lap 1
        current_lap = 1
        start_t = 0.0
        
        # Beacons mark the END of a lap
        for marker in beacon_markers:
            mask = (df['Time'] >= start_t) & (df['Time'] < marker)
            df.loc[mask, 'lap'] = current_lap
            start_t = marker
            current_lap += 1
            
        # 4. Standardize Names for the AI
        # The AI expects lowercase: 'rpm', 'speed_mph', 'lat_g', 'long_g'
        rename_map = {
            'RPM': 'rpm',
            'GPS Speed': 'speed_mph',
            'Steering Angle': 'steer',
            'GPS LatAcc': 'lat_g',
            'GPS LonAcc': 'long_g'
        }
        df = df.rename(columns=rename_map)
        
        # Select final columns
        final_cols = ['Time', 'lap', 'rpm', 'speed_mph', 'steer', 'lat_g', 'long_g']
        final_cols = [c for c in final_cols if c in df.columns]
        
        return df[final_cols], "Success (Single File)"
        
    except Exception as e:
        return None, f"Error processing single file: {e}"

def process_batch_files(folder_path):
    """Original logic for stitching RPM.csv + _GPS.csv"""
    print("🔄 No single file found. Attempting batch stitch...")
    files = {'rpm': 'RPM.csv', 'gps': '_GPS.csv', 'steer': 'Steering Angle.csv', 'gyro': 'GyroZ.csv'}
    data_frames = {}
    
    # Load available files
    for key, filename in files.items():
        path = os.path.join(folder_path, filename)
        if os.path.exists(path):
            try:
                # GPS usually has headers on row 0, others on row 1
                skip = [1] if key != 'gps' else None
                df = pd.read_csv(path, skiprows=skip)
                df.columns = [c.strip().lower() for c in df.columns]
                if 'time' in df.columns:
                    df['time'] = pd.to_numeric(df['time'], errors='coerce')
                    df = df.dropna(subset=['time']).sort_values('time')
                    data_frames[key] = df
            except: pass

    if 'rpm' not in data_frames:
        return None, "RPM file missing."

    # Merge
    master = data_frames['rpm'].rename(columns={'value': 'rpm'})[['time', 'rpm']]
    if 'gps' in data_frames:
        # Find speed column
        cols = data_frames['gps'].columns
        speed_col = next((c for c in cols if 'speed' in c), 'speed')
        gps_cols = ['time', speed_col, 'lat', 'lon']
        # Rename to 'speed' for consistency before merge
        data_frames['gps'].rename(columns={speed_col: 'speed'}, inplace=True)
        # Only merge existing columns
        gps_cols = [c for c in gps_cols if c in data_frames['gps'].columns]
        master = pd.merge_asof(master, data_frames['gps'][gps_cols], on='time', direction='nearest')

    if 'steer' in data_frames:
        steer = data_frames['steer'].rename(columns={'value': 'steer'})[['time', 'steer']]
        master = pd.merge_asof(master, steer, on='time', direction='nearest')
        
    if 'gyro' in data_frames:
        gyro = data_frames['gyro'].rename(columns={'value': 'yaw_rate'})[['time', 'yaw_rate']]
        master = pd.merge_asof(master, gyro, on='time', direction='nearest')

    # Physics
    master = master.fillna(0)
    if 'speed' in master.columns:
        master['speed_mph'] = master['speed'] * 2.237
    if 'yaw_rate' in master.columns and 'speed' in master.columns:
        master['lat_g'] = (master['speed'] * np.radians(master['yaw_rate'])) / 9.81 * -1
        
    # Default Lap 0 if no summary file found (Batch mode needs summary file for laps)
    master['lap'] = 0
    
    return master, "Success (Batch)"