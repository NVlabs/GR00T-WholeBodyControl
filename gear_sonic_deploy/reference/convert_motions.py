#!/usr/bin/env python3
"""
Convert G1 motion capture data from joblib pickle to C++ readable formats
Converts motion sequences with joint positions, velocities, and full body kinematics
"""

import sys
import os

def convert_motion_data(pkl_file, base_output_dir=None):
    """Convert the motion pickle file to C++ readable formats"""
    
    # Create organized folder structure: reference/{pkl_name}/
    pkl_name = os.path.splitext(os.path.basename(pkl_file))[0]  # Extract filename without extension
    
    if base_output_dir is None:
        # Default: put in reference folder structure
        base_output_dir = os.path.join(os.path.dirname(pkl_file), pkl_name)
    
    print(f"Converting motion data from: {pkl_file}")
    print(f"Output directory structure: {base_output_dir}/")
    
    # Load the data
    try:
        import joblib
        data = joblib.load(pkl_file)
        print("✓ Successfully loaded with joblib")
    except ImportError:
        print("✗ joblib not available, trying pickle...")
        try:
            import pickle
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            print("✓ Successfully loaded with pickle")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            return False
    except Exception as e:
        print(f"✗ Failed to load with joblib: {e}")
        return False
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"\nFound {len(data)} motion sequences:")
    for motion_name in data.keys():
        print(f"  - {motion_name}")
    
    # Convert each motion sequence
    success_count = 0
    for motion_name, motion_data in data.items():
        print(f"\nProcessing: {motion_name}")
        
        # Create individual folder for this motion
        motion_output_dir = os.path.join(base_output_dir, motion_name)
        print(f"Creating individual folder for this motion: {motion_output_dir}")
        os.makedirs(motion_output_dir, exist_ok=True)
        
        if convert_single_motion(motion_name, motion_data, motion_output_dir):
            success_count += 1
    
    # Create summary file in base directory
    create_summary_file(data, base_output_dir)
    
    print(f"\n✓ Successfully converted {success_count}/{len(data)} motions")
    print(f"Output files saved to: {base_output_dir}/")
    
    # Extract counts from first motion for summary
    joint_count = None
    body_count = None
    if data:
        first_motion = next(iter(data.values()))
        joint_count = first_motion['joint_pos'].shape[1]
        body_count = first_motion['body_pos_w'].shape[1]
    
    return success_count > 0, len(data), joint_count, body_count

def convert_single_motion(motion_name, motion_data, output_dir):
    """Convert a single motion sequence to various formats"""
    
    try:
        # Extract ALL available data arrays
        joint_pos = motion_data['joint_pos']  # Shape: (timesteps, 29)
        joint_vel = motion_data['joint_vel']  # Shape: (timesteps, 29)  
        body_pos_w = motion_data['body_pos_w']  # Shape: (timesteps, 14, 3)
        body_quat_w = motion_data['body_quat_w']  # Shape: (timesteps, 14, 4) 
        body_lin_vel_w = motion_data['body_lin_vel_w']  # Shape: (timesteps, 14, 3)
        body_ang_vel_w = motion_data['body_ang_vel_w']  # Shape: (timesteps, 14, 3)
        
        timesteps = joint_pos.shape[0]
        print(f"  Timesteps: {timesteps}, Joints: {joint_pos.shape[1]}, Body parts: {body_pos_w.shape[1]}")
        
        # 1. Save joint data as CSV
        joint_pos_file = os.path.join(output_dir, "joint_pos.csv")
        save_array_as_csv(joint_pos, joint_pos_file, 
                         [f"joint_{i}" for i in range(joint_pos.shape[1])])
        
        joint_vel_file = os.path.join(output_dir, "joint_vel.csv")
        save_array_as_csv(joint_vel, joint_vel_file,
                         [f"joint_vel_{i}" for i in range(joint_vel.shape[1])])
        
        # 2. Save body position data (reshape to 2D for CSV)
        body_pos_reshaped = body_pos_w.reshape(timesteps, -1)  # (timesteps, 14*3)
        body_pos_file = os.path.join(output_dir, "body_pos.csv")
        body_pos_headers = [f"body_{i//3}_{'xyz'[i%3]}" for i in range(body_pos_reshaped.shape[1])]
        save_array_as_csv(body_pos_reshaped, body_pos_file, body_pos_headers)
        
        # 3. Save body quaternion data (reshape to 2D for CSV)
        body_quat_reshaped = body_quat_w.reshape(timesteps, -1)  # (timesteps, 14*4)  
        body_quat_file = os.path.join(output_dir, "body_quat.csv")
        body_quat_headers = [f"body_{i//4}_{'wxyz'[i%4]}" for i in range(body_quat_reshaped.shape[1])]
        save_array_as_csv(body_quat_reshaped, body_quat_file, body_quat_headers)
        
        # 4. Save body linear velocity data
        body_lin_vel_reshaped = body_lin_vel_w.reshape(timesteps, -1)  # (timesteps, 14*3)
        body_lin_vel_file = os.path.join(output_dir, "body_lin_vel.csv")
        body_lin_vel_headers = [f"body_{i//3}_vel_{'xyz'[i%3]}" for i in range(body_lin_vel_reshaped.shape[1])]
        save_array_as_csv(body_lin_vel_reshaped, body_lin_vel_file, body_lin_vel_headers)
        
        # 5. Save body angular velocity data  
        body_ang_vel_reshaped = body_ang_vel_w.reshape(timesteps, -1)  # (timesteps, 14*3)
        body_ang_vel_file = os.path.join(output_dir, "body_ang_vel.csv")
        body_ang_vel_headers = [f"body_{i//3}_angvel_{'xyz'[i%3]}" for i in range(body_ang_vel_reshaped.shape[1])]
        save_array_as_csv(body_ang_vel_reshaped, body_ang_vel_file, body_ang_vel_headers)
        
        # 6. Save metadata
        metadata_file = os.path.join(output_dir, "metadata.txt")
        save_metadata(motion_name, motion_data, metadata_file)
        
        # 7. Save detailed info
        info_file = os.path.join(output_dir, "info.txt")
        save_motion_info(motion_name, motion_data, info_file)
        
        print(f"  ✓ Saved 7 files for {motion_name} (joints + full body kinematics)")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {motion_name}: {e}")
        return False

def save_array_as_csv(array, filename, headers=None):
    """Save numpy array as CSV file"""
    import numpy as np
    
    with open(filename, 'w') as f:
        # Write header
        if headers:
            f.write(",".join(headers) + "\n")
        else:
            f.write(",".join([f"col_{i}" for i in range(array.shape[1])]) + "\n")
        
        # Write data
        for row in array:
            f.write(",".join([f"{val:.6f}" for val in row]) + "\n")


def save_metadata(motion_name, motion_data, filename):
    """Save metadata and indices information"""
    
    with open(filename, 'w') as f:
        f.write(f"Metadata for: {motion_name}\n")
        f.write("=" * 30 + "\n\n")
        
        # Save body indexes if available
        if '_body_indexes' in motion_data:
            f.write("Body part indexes:\n")
            f.write(f"{motion_data['_body_indexes']}\n\n")
        
        # Save total timestep count
        if 'time_step_total' in motion_data:
            f.write(f"Total timesteps: {motion_data['time_step_total']}\n\n")
        
        # Data summary
        f.write("Data arrays summary:\n")
        for key, value in motion_data.items():
            if hasattr(value, 'shape'):
                f.write(f"  {key}: {value.shape} ({value.dtype})\n")

def save_motion_info(motion_name, motion_data, filename):
    """Save detailed motion information"""
    
    with open(filename, 'w') as f:
        f.write(f"Motion Information: {motion_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in motion_data.items():
            f.write(f"{key}:\n")
            if hasattr(value, 'shape'):
                f.write(f"  Shape: {value.shape}\n")
                f.write(f"  Dtype: {value.dtype}\n")
                if value.size > 0:
                    flat_vals = value.flatten()
                    f.write(f"  Range: [{flat_vals.min():.3f}, {flat_vals.max():.3f}]\n")
                    f.write(f"  Sample: {flat_vals[:5]}\n")
            else:
                f.write(f"  Value: {value}\n")
            f.write("\n")

def create_summary_file(data, output_dir):
    """Create a summary file with all motion information"""
    
    summary_file = os.path.join(output_dir, "motion_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("G1 Motion Capture Data Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total motion sequences: {len(data)}\n\n")
        
        
        # Detailed motion list
        f.write("Detailed motion list:\n")
        for motion_name, motion_data in data.items():
            joint_pos = motion_data['joint_pos']
            f.write(f"  {motion_name}:\n")
            f.write(f"    Timesteps: {joint_pos.shape[0]}\n") 
            f.write(f"    Joints: {joint_pos.shape[1]}\n")
            f.write(f"    Body parts: {motion_data['body_pos_w'].shape[1]}\n")
            f.write("\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_motions.py <pkl_file> [output_base_dir]")
        print("Examples:")
        print("  python3 convert_motions.py bones_072925_test.pkl")
        print("  python3 convert_motions.py bones_072925_test.pkl custom_output/")
        print("")
        print("Default output structure: reference/{pkl_name}/{motion_name}/")
        return
    
    pkl_file = sys.argv[1]
    output_base_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(pkl_file):
        print(f"Error: File not found: {pkl_file}")
        return
    
    # Extract pkl name for output messages
    pkl_name = os.path.splitext(os.path.basename(pkl_file))[0]
    
    print("G1 Motion Data Converter")
    print("========================")
    
    success, motion_count, joint_count, body_count = convert_motion_data(pkl_file, output_base_dir)
    if success:
        print("\n✓ Conversion completed successfully!")
        print("\nExtracted data for each motion:")
        print(f"- Joint positions & velocities ({joint_count} joints)")  
        print(f"- Body positions in world coordinates ({body_count} body parts)")
        print("- Body orientations (quaternions)")
        print("- Body linear & angular velocities")
        print("- Metadata and body part indices")
        print("\nNext steps:")
        print(f"1. Build C++ reader: make motion_data_reader")
        print(f"2. Test reading: ./bin/motion_data_reader reference/{pkl_name}/")
        print("3. Use full kinematic data in your G1 control programs")
        print("\nFile structure created:")
        print(f"reference/{pkl_name}/")
        print("├── [motion_name_1]/")
        print("│   ├── joint_pos.csv")
        print("│   ├── joint_vel.csv")
        print("│   ├── body_pos.csv")
        print("│   ├── body_quat.csv")
        print("│   ├── body_lin_vel.csv")
        print("│   ├── body_ang_vel.csv")
        print("│   ├── metadata.txt")
        print("│   └── info.txt")
        print("├── [motion_name_2]/")
        print("├── motion_summary.txt")
        print(f"└── ... ({motion_count} motion folders total)")
    else:
        print("\n✗ Conversion failed")

if __name__ == "__main__":
    main()
