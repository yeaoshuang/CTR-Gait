import numpy as np
import pickle
import os

# BPM Complete Data Generation Tool
# Generate bone and motion data from joint data

# --------------------------
# 1. Configuration Parameters and Paths
# --------------------------
# Modify paths to match current project structure
BPM_DATA_DIR = r"d:\recognition-of-emotion\CTR_GCNgait\bpm_data"  # BPM data directory
OUTPUT_DIR = r"d:\recognition-of-emotion\CTR_GCNgait\bpm_data"  # Output file save directory

# Bone connection definitions
bone_connections = [
    (0, 10), (10, 11), (11, 12),  # Left leg: root → hip → knee → foot
    (0, 13), (13, 14), (14, 15),  # Right leg: root → hip → knee → foot
    (0, 1),  (1, 2),   (2, 3),    # Torso: root → spine → neck → head
    (1, 4),  (4, 5),   (5, 6),    # Left arm: spine → shoulder → elbow → hand
    (1, 7),  (7, 8),   (8, 9)     # Right arm: spine → shoulder → elbow → hand
]

# --------------------------
# 2. Data Generation Functions
# --------------------------
def compute_bone_data(joint_data):
    """Calculate bone connection data - improved version, consistent with current model operations
    
    Design Description:
    1. Ensure all 15 bone vectors have specific values: calculate bone vectors for each bone connection, 
       stored in the first 15 dimensions
    2. Add a zero vector at the end to form 16 dimensions: the 16th dimension (index 15) remains as zero vector
    
    Benefits of this approach:
    - Ensure all 15 bone connections have corresponding bone vector representations
    - Keep data dimensions consistent with joint point dimensions (16 dimensions), facilitating subsequent processing
    - Zero vectors do not affect model calculation results
    
    Bone vector calculation method:
    For each bone connection (v1, v2), calculate bone vector as: joint_data[v1] - joint_data[v2]
    This is consistent with the generate_bone_data method in utils/data_loader.py
    
    Args:
        joint_data: Original joint data with shape (N, C, T, V, M)
    
    Returns:
        bone_data: Bone connection data with shape (N, C, T, V, M)
        Where the first 15 dimensions contain specific bone vectors, and the 16th dimension is a zero vector
    """
    print("Calculating bone connection data...")
    N, C, T, V, M = joint_data.shape  # [Number of samples, number of coordinates, time frames, number of joints, number of people]
    bone_data = np.zeros_like(joint_data)  # Keep the same shape as input (N, C, T, V, M)
    
    # Calculate bone vectors for each bone connection, stored in the first 15 dimensions
    for i, (v1, v2) in enumerate(bone_connections):
        # Calculate vector between two joints: v1 -> v2
        bone_data[:, :, :, i, :] = joint_data[:, :, :, v1, :] - joint_data[:, :, :, v2, :]
    
    # The 16th dimension (index 15) remains as zero vector
    # This ensures 15 bone vectors have specific values, then add a zero vector at the end to form 16 dimensions
    
    print(f"Bone data shape: {bone_data.shape}")
    return bone_data

def compute_motion_energy(joint_data):
    """Calculate motion data (joint velocity only, no energy)
    
    Args:
        joint_data: Original joint data with shape (N, C, T, V, M)
    
    Returns:
        motion: Motion data with shape (N, 3, T, V, M)
            Contains only joint velocities (x, y, z)
    """
    print("Calculating motion data...")
    N, C, T, V, M = joint_data.shape  # [Number of samples, number of coordinates, time frames, number of joints, number of people]
    
    # Create motion data, set first frame to all zeros
    # Shape is (N, C, T, V, M) = (N, 3, T, V, M)
    motion = np.zeros((N, C, T, V, M), dtype=np.float32)
    
    # Calculate velocity (difference between current frame and previous frame)
    velocity = joint_data[:, :, 1:, :, :] - joint_data[:, :, :-1, :, :]  # (N, 3, 47, 16, 1)
    
    # Put velocity data into frames 2 to 48 of motion (indices 1-47)
    motion[:, :, 1:, :, :] = velocity
    
    # First frame (index 0) remains all zeros
    print(f"Motion data shape: {motion.shape}")
    return motion

def generate_complete_data():
    """Generate complete BPM data (joint, bone, motion)"""
    print("Starting to generate complete BPM data...")
    
    # Process training data
    print("\nProcessing training data...")
    train_joint_path = os.path.join(BPM_DATA_DIR, 'train_joint.npy')
    train_label_path = os.path.join(BPM_DATA_DIR, 'train_label.pkl')
    
    if os.path.exists(train_joint_path):
        # Load joint data
        train_joint = np.load(train_joint_path)
        print(f"Training joint data shape: {train_joint.shape}")
        
        # Generate bone data
        train_bone = compute_bone_data(train_joint)
        np.save(os.path.join(OUTPUT_DIR, 'train_bone.npy'), train_bone)
        
        # Generate motion data
        train_motion = compute_motion_energy(train_joint)
        np.save(os.path.join(OUTPUT_DIR, 'train_motion.npy'), train_motion)
        
        # Copy label file as joint.pkl format (if needed)
        if os.path.exists(train_label_path):
            # Read original labels
            with open(train_label_path, 'rb') as f:
                train_labels = pickle.load(f)
            
            # Save label file (shared by all modalities)
            with open(os.path.join(OUTPUT_DIR, 'train_label.pkl'), 'wb') as f:
                pickle.dump(train_labels, f)
        
        print("Training data generation completed!")
    
    # Process test data
    print("\nProcessing test data...")
    test_joint_path = os.path.join(BPM_DATA_DIR, 'test_joint.npy')
    test_label_path = os.path.join(BPM_DATA_DIR, 'test_label.pkl')
    
    if os.path.exists(test_joint_path):
        # Load joint data
        test_joint = np.load(test_joint_path)
        print(f"Test joint data shape: {test_joint.shape}")
        
        # Generate bone data
        test_bone = compute_bone_data(test_joint)
        np.save(os.path.join(OUTPUT_DIR, 'test_bone.npy'), test_bone)
        
        # Generate motion data
        test_motion = compute_motion_energy(test_joint)
        np.save(os.path.join(OUTPUT_DIR, 'test_motion.npy'), test_motion)
        
        # Copy label file as joint.pkl format (if needed)
        if os.path.exists(test_label_path):
            # Read original labels
            with open(test_label_path, 'rb') as f:
                test_labels = pickle.load(f)
            
            # Save label file (shared by all modalities)
            with open(os.path.join(OUTPUT_DIR, 'test_label.pkl'), 'wb') as f:
                pickle.dump(test_labels, f)
        
        print("Test data generation completed!")
    
    print("\nComplete BPM data generation finished!")

# --------------------------
# 3. Validate Generated Data
# --------------------------
def validate_generated_data():
    """Validate generated data"""
    print("Validating generated data...")
    
    required_files = [
        'train_bone.npy', 'train_motion.npy',
        'test_bone.npy', 'test_motion.npy',
        'train_label.pkl', 'test_label.pkl'  # Label files (shared by all modalities)
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    
    # Validate data shapes
    try:
        train_bone = np.load(os.path.join(OUTPUT_DIR, 'train_bone.npy'))
        train_motion = np.load(os.path.join(OUTPUT_DIR, 'train_motion.npy'))
        
        print(f"Training set bone data shape: {train_bone.shape}")
        print(f"Training set motion data shape: {train_motion.shape}")
        
        test_bone = np.load(os.path.join(OUTPUT_DIR, 'test_bone.npy'))
        test_motion = np.load(os.path.join(OUTPUT_DIR, 'test_motion.npy'))
        
        print(f"Test set bone data shape: {test_bone.shape}")
        print(f"Test set motion data shape: {test_motion.shape}")
        
        # Validate labels
        with open(os.path.join(OUTPUT_DIR, 'train_label.pkl'), 'rb') as f:
            train_labels = pickle.load(f)
        with open(os.path.join(OUTPUT_DIR, 'test_label.pkl'), 'rb') as f:
            test_labels = pickle.load(f)
        
        # Get actual label list (second element of tuple)
        train_label_list = train_labels[1] if isinstance(train_labels, tuple) else train_labels
        test_label_list = test_labels[1] if isinstance(test_labels, tuple) else test_labels
        
        # Calculate number of unique labels
        unique_train_labels = len(set(train_label_list))
        unique_test_labels = len(set(test_label_list))
        
        print(f"Training set label count: {unique_train_labels} (total samples: {len(train_label_list)})")
        print(f"Test set label count: {unique_test_labels} (total samples: {len(test_label_list)})")
        
        # Show label distribution
        train_label_dist = {label: train_label_list.count(label) for label in set(train_label_list)}
        test_label_dist = {label: test_label_list.count(label) for label in set(test_label_list)}
        print(f"Training set label distribution: {train_label_dist}")
        print(f"Test set label distribution: {test_label_dist}")
        
        print("Data validation passed!")
        return True
    except Exception as e:
        print(f"Data validation failed: {str(e)}")
        return False

# --------------------------
# 4. Main Function
# --------------------------
def main():
    """Main function"""
    print("=" * 50)
    print("BPM Complete Data Generation Tool")
    print("=" * 50)
    
    # Generate complete data
    generate_complete_data()
    
    # Validate generated data
    if validate_generated_data():
        print("\n" + "=" * 50)
        print("BPM Complete Data Generation Completed!")
        print("=" * 50)
        print("\nNow you can:")
        print("1. View usage instructions: BPM_DATA_USAGE.md")
        print("2. Start training: python main_adapted.py --config config/EGait_journal/train_bpm.yaml")
        print("3. Test model: python main_adapted.py --config config/EGait_journal/test_bpm.yaml")
    else:
        print("\nData generation or validation failed!")

if __name__ == "__main__":
    main()