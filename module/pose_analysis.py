def calculate_box_ratio(box):
    """
    Calculate ratio between top edge (AB) and side edge (BC) of bounding box
    Returns ratio and posture state with like_fall levels 1-4
    """
    x1, y1, x2, y2 = box
    
    # Calculate edges
    AB = abs(x2 - x1)  # width (top edge)
    BC = abs(y2 - y1)  # height (side edge)
    
    ratio = AB / BC if BC != 0 else 0
    
    # Determine posture state with like_fall levels
    if AB < BC and ratio < 0.5:
        state = "stand"
    elif AB > BC and ratio > 1.2:
        state = "lie"
    else:
        # Define like_fall levels based on how close ratio is to 1.2
        if ratio < 0.7:
            state = "like_fall_1"  # Closest to standing
        elif ratio < 0.9:
            state = "like_fall_2"
        elif ratio < 1.1:
            state = "like_fall_3"
        else:
            state = "like_fall_4"  # Closest to lying
        
    return ratio, state

def get_box_corners(box):
    """
    Returns the four corners of the bounding box
    A,B are top corners, C,D are bottom corners
    """
    x1, y1, x2, y2 = box
    
    A = (x1, y1)  # Top-left
    B = (x2, y1)  # Top-right
    C = (x2, y2)  # Bottom-right
    D = (x1, y2)  # Bottom-left
    
    return A, B, C, D
