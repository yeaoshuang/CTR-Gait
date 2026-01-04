import numpy as np

# Dataset graph structure definition (joint points)
num_node = 16
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 11), (11, 12), (12, 13),  # Left leg
    (1, 14), (14, 15), (15, 16),  # Right leg
    (1, 2),  (2, 3),   (3, 4),    # Torso
    (2, 5),  (5, 6),   (6, 7),    # Left arm
    (2, 8),  (8, 9),   (9, 10)    # Right arm
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# Body part definition (joint points)
parts = [
    np.array([5, 6, 7]) - 1,      # Left arm
    np.array([8, 9, 10]) - 1,     # Right arm
    np.array([11, 12, 13]) - 1,   # Left leg
    np.array([14, 15, 16]) - 1,   # Right leg
    np.array([1, 2, 3, 4]) - 1    # Torso
]

# Joint connection definition
connect_joint = np.array([1,1,2,3,2,5,6,2,8,9,1,11,12,1,14,15]) - 1

# ================================
# Bone data specific graph structure definition
# ================================

# Bone data node count (15 bone vectors)
bone_num_node = 15

# Bone connection definition (consistent with bone_connections in generate_bpm_complete_data.py)
# Note: The indices here are bone vector indices (0-14), not joint point indices
bone_connections = [
    (0, 10), (10, 11), (11, 12),  # Left leg: root → hip → knee → foot
    (0, 13), (13, 14), (14, 15),  # Right leg: root → hip → knee → foot
    (0, 1),  (1, 2),   (2, 3),    # Torso: root → spine → neck → head
    (1, 4),  (4, 5),   (5, 6),    # Left arm: spine → shoulder → elbow → hand
    (1, 7),  (7, 8),   (8, 9)     # Right arm: spine → shoulder → elbow → hand
]

# Bone self-connection
bone_self_link = [(i, i) for i in range(bone_num_node)]

# Bone connection relationship (based on shared joints between bone vectors)
# Two bone vectors are considered connected if they share a joint point
bone_inward = []
for i in range(bone_num_node):
    for j in range(i+1, bone_num_node):
        # Check if bone i and bone j share joint points
        joint_i = set(bone_connections[i])
        joint_j = set(bone_connections[j])
        if joint_i & joint_j:  # If intersection exists, they share joint points
            bone_inward.append((i, j))

bone_outward = [(j, i) for (i, j) in bone_inward]
bone_neighbor = bone_inward + bone_outward

# Bone body part definition - redefined based on actual bone_connections indices
# Note: The indices here are bone vector indices (0-14), not joint point indices
bone_parts = [
    np.array([0, 1, 2]),          # Left leg (indices 0,1,2 correspond to 3 bones of left leg)
    np.array([3, 4, 5]),          # Right leg (indices 3,4,5 correspond to 3 bones of right leg)
    np.array([6, 7, 8]),          # Torso (indices 6,7,8 correspond to 3 bones of torso)
    np.array([9, 10, 11]),        # Left arm (indices 9,10,11 correspond to 3 bones of left arm)
    np.array([12, 13, 14])        # Right arm (indices 12,13,14 correspond to 3 bones of right arm)
]

# Bone to joint mapping (for visualization or analysis)
# Note: The indices here are bone vector indices (0-14), values are joint point indices
bone_to_joints = {
    0: (0, 10),   # Left leg: root → hip
    1: (10, 11),  # Left leg: hip → knee
    2: (11, 12),  # Left leg: knee → foot
    3: (0, 13),   # Right leg: root → hip
    4: (13, 14),  # Right leg: hip → knee
    5: (14, 15),  # Right leg: knee → foot
    6: (0, 1),    # Torso: root → spine
    7: (1, 2),    # Torso: spine → neck
    8: (2, 3),    # Torso: neck → head
    9: (1, 4),    # Left arm: spine → shoulder
    10: (4, 5),   # Left arm: shoulder → elbow
    11: (5, 6),   # Left arm: elbow → hand
    12: (1, 7),   # Right arm: spine → shoulder
    13: (7, 8),   # Right arm: shoulder → elbow
    14: (8, 9)    # Right arm: elbow → hand
}

# Bone vector name mapping (for interpretation)
bone_names = {
    0: "Left leg-root to hip",
    1: "Left leg-hip to knee",
    2: "Left leg-knee to foot",
    3: "Right leg-root to hip",
    4: "Right leg-hip to knee",
    5: "Right leg-knee to foot",
    6: "Torso-root to spine",
    7: "Torso-spine to neck",
    8: "Torso-neck to head",
    9: "Left arm-spine to shoulder",
    10: "Left arm-shoulder to elbow",
    11: "Left arm-elbow to hand",
    12: "Right arm-spine to shoulder",
    13: "Right arm-shoulder to elbow",
    14: "Right arm-elbow to hand"
}

class Graph:
    """Dataset graph structure class"""
    
    def __init__(self, labeling_mode='spatial', data_type='joint'):
        """
        Initialize graph structure
        
        Args:
            labeling_mode: Labeling mode, default 'spatial'
            data_type: Data type, 'joint' for joint data, 'bone' for bone data
        """
        self.data_type = data_type
        
        if data_type == 'joint':
            # Graph structure for joint data
            self.num_node = num_node
            self.self_link = self_link
            self.inward = inward
            self.outward = outward
            self.neighbor = neighbor
            self.parts = parts
            self.connect_joint = connect_joint
        elif data_type == 'bone':
            # Graph structure for bone data
            # Note: Although bone data has 15 valid bone vectors, the data shape is (N, C, T, 16, M)
            # The 16th dimension (index 15) is a zero vector to maintain consistent shape with joint data
            self.num_node = num_node  # Use 16, consistent with data shape
            self.bone_num_node = bone_num_node  # Actual number of bone vectors
            
            # Create self-connections for 16 nodes (including the 16th zero vector node)
            self.self_link = [(i, i) for i in range(self.num_node)]
            
            # Connection relationships for the first 15 nodes
            self.bone_inward = bone_inward
            self.bone_outward = bone_outward
            self.bone_neighbor = bone_inward + bone_outward
            
            # Extend connection relationships to 16 nodes (16th node not connected to other nodes)
            self.inward = bone_inward
            self.outward = bone_outward
            self.neighbor = bone_inward + bone_outward
            
            # Body part definition (only includes first 15 nodes)
            self.bone_parts = bone_parts
            self.parts = bone_parts
            
            # Bone to joint mapping
            self.bone_to_joints = bone_to_joints
            self.bone_names = bone_names
            
            self.connect_joint = None  # Bone data doesn't need connect_joint
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        """Get adjacency matrix"""
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = self.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A

    def get_spatial_graph(self, num_node, self_link, inward, outward):
        """Get spatial graph"""
        I = self.edge2mat(self_link, num_node)
        In = self.normalize_digraph(self.edge2mat(inward, num_node))
        Out = self.normalize_digraph(self.edge2mat(outward, num_node))
        A = np.stack((I, In, Out))
        
        # For bone data, special handling is needed for the 16th node (index 15)
        if self.data_type == 'bone':
            # Ensure the 16th node (index 15) only has self-connection, not connected to other nodes
            A[0, :, 15] = 0  # Clear self-connection of the 16th node
            A[0, 15, :] = 0
            A[0, 15, 15] = 1  # Keep only self-connection
            
            # Clear connections between the 16th node and other nodes
            A[1, :, 15] = 0  # Clear inward connections
            A[1, 15, :] = 0
            A[2, :, 15] = 0  # Clear outward connections
            A[2, 15, :] = 0
            
        return A

    def edge2mat(self, link, num_node):
        """Edge to matrix conversion"""
        A = np.zeros((num_node, num_node))
        for i, j in link:
            A[j, i] = 1
        return A

    def normalize_digraph(self, A):
        """Directed graph normalization"""
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    def get_hop_distance(self):
        """Get hop distance"""
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.neighbor:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(10 + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(10, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_adjacency(self):
        """Get adjacency matrix"""
        hop_dis = self.get_hop_distance()
        valid_hop = range(0, 10 + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = np.zeros((self.num_node, self.num_node))
        Dl = np.sum(adjacency, 0)
        for i in range(len(Dl)):
            if Dl[i] > 0:
                normalize_adjacency[i, i] = Dl[i] ** (-1)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A