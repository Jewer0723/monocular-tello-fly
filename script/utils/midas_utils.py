import numpy as np

# =====================
# 深度處理函數
# =====================
def get_depth_regions(depth_map):
    """獲取各區域深度值"""
    h, w = depth_map.shape

    # 中心區域（正前方）
    center_height = h // 3
    center_width = w // 3
    center_top = h // 2 - center_height // 2
    center_left = w // 2 - center_width // 2

    center = depth_map[center_top:center_top + center_height,
             center_left:center_left + center_width]

    # 左側區域
    left = depth_map[center_top:center_top + center_height, :w // 4]

    # 右側區域
    right = depth_map[center_top:center_top + center_height, 3 * w // 4:]

    # 計算中位數（減少噪聲影響）
    center_val = np.median(center) if center.size > 0 else 0.5
    left_val = np.median(left) if left.size > 0 else 0.5
    right_val = np.median(right) if right.size > 0 else 0.5

    return center_val, left_val, right_val