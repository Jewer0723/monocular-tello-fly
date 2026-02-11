import pygame
import time

# =====================
# pygame 初始化
# =====================
pygame.init()
pygame.display.set_mode((300, 200))
pygame.display.set_caption("Tello Keyboard Control")

# =====================
# 鍵盤控制
# =====================
def get_keyboard_control(tello):
    """
    鍵盤控制（符合你指定的鍵位）
    回傳:
        manual_active (bool)
        lr, fb, ud, yv
    """
    lr = fb = ud = yv = 0
    manual_active = False

    pygame.event.pump()
    keys = pygame.key.get_pressed()

    SPEED = 30
    YAW_SPEED = 50
    UD_SPEED = 40

    # ---------- 上升 / 下降 ----------
    if keys[pygame.K_w]:
        ud = UD_SPEED
        manual_active = True
    if keys[pygame.K_s]:
        ud = -UD_SPEED
        manual_active = True

    # ---------- 左轉 / 右轉 ----------
    if keys[pygame.K_a]:
        yv = -YAW_SPEED
        manual_active = True
    if keys[pygame.K_d]:
        yv = YAW_SPEED
        manual_active = True

    # ---------- 前進 / 後退 ----------
    if keys[pygame.K_UP]:
        fb = SPEED
        manual_active = True
    if keys[pygame.K_DOWN]:
        fb = -SPEED
        manual_active = True

    # ---------- 左右平移 ----------
    if keys[pygame.K_LEFT]:
        lr = -SPEED
        manual_active = True
    if keys[pygame.K_RIGHT]:
        lr = SPEED
        manual_active = True

    # ---------- 懸停 ----------
    if keys[pygame.K_SPACE]:
        lr = fb = ud = yv = 0
        manual_active = True

    # ---------- 起飛 ----------
    if keys[pygame.K_t]:
        print("⌨️ 鍵盤起飛")
        tello.takeoff()
        time.sleep(1)

    # ---------- 降落 ----------
    if keys[pygame.K_l]:
        print("⌨️ 鍵盤降落")
        tello.land()
        time.sleep(1)

    # ---------- ESC：結束程式 ----------
    if keys[pygame.K_ESCAPE]:
        print("⌨️ 使用者停止程式")
        pygame.quit()
        return True, 0, 0, 0, 0, True  # 最後一個 True 表示要退出

    return manual_active, lr, fb, ud, yv, False