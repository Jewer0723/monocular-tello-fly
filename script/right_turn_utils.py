import time

# =====================
# 簡單右轉避障控制器
# =====================
class RightTurnAvoider:
    def __init__(self):
        self.state = "FORWARD"  # 狀態: FORWARD, TURNING
        self.turn_start_time = 0
        self.turn_direction = 1  # 1=右轉，-1=左轉（我們固定用右轉）
        self.obstacle_count = 0
        self.last_depth = 0.5

    def update_state(self, center_depth, OBSTACLE_THRESHOLD, TURN_DURATION, CLEAR_THRESHOLD):
        """更新狀態機"""
        current_time = time.time()

        # 狀態轉換邏輯
        if self.state == "FORWARD":
            # 檢查是否需要轉向
            if center_depth > OBSTACLE_THRESHOLD:
                self.state = "TURNING"
                self.turn_start_time = current_time
                self.obstacle_count += 1
                print(f"🚨 發現障礙物！深度: {center_depth:.3f} > {OBSTACLE_THRESHOLD}")
                print(f"開始向右轉...")
                return "TURNING"
            else:
                return "FORWARD"

        elif self.state == "TURNING":
            # 檢查是否轉夠了
            turn_elapsed = current_time - self.turn_start_time

            if turn_elapsed >= TURN_DURATION:
                # 檢查轉向後是否安全
                if center_depth < CLEAR_THRESHOLD:
                    self.state = "FORWARD"
                    print(f"✅ 轉向完成，前方安全！深度: {center_depth:.3f}")
                    return "FORWARD"
                else:
                    # 還是不安全，繼續轉
                    print(f"⚠️ 轉向後仍不安全，繼續轉... 深度: {center_depth:.3f}")
                    self.turn_start_time = current_time
                    return "TURNING"
            else:
                # 還在轉向中
                return "TURNING"

        return self.state

    def get_control(self, state, BASE_FORWARD, TURN_SPEED):
        """根據狀態返回控制指令"""
        fbv = 0  # 前後速度
        yv = 0  # 左右轉向

        if state == "FORWARD":
            fbv = BASE_FORWARD
            yv = 0
        elif state == "TURNING":
            fbv = 0  # 原地轉向
            yv = TURN_SPEED  # 固定向右轉

        return fbv, yv