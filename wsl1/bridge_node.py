#!/usr/bin/env python3
"""
bridge_node.py — WSL1 Ubuntu 20.04 + ROS Noetic
改良版：ORB 失去後不重置，繼續用 DR 累積，恢復時從上次位置繼續
"""
import socket, json, math, threading, time
import rospy
import tf2_ros
from geometry_msgs.msg import Point, TransformStamped, Quaternion, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String

UDP_PORT      = 9999
ORB_CORR_PORT = 9997
FRAME_ID      = "world"
SCALE         = 0.25   # cm → m
ORB_SCALE     = 1.0    # ORB 不放大
MAX_PATH      = 8000
ORB_TIMEOUT   = 1.5   # 超過此秒數沒收到 pose → 視為 ORB lost

# 新增：ORB 失去後的最大等待時間（超過此時間才重置）
ORB_MAX_LOST_TIME = 5.0

def yaw_to_quat(yaw_deg):
    half = math.radians(-yaw_deg) / 2.0
    q = Quaternion()
    q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(half), math.cos(half)
    return q

def dr_to_rviz(x_cm, z_cm):
    """只用水平面 x/z，rviz Z 固定 0，消除高度噪音導致的垂直漂移"""
    return z_cm * SCALE, -x_cm * SCALE, 0.0

def orb_to_local(p):
    return p.z * ORB_SCALE, -p.x * ORB_SCALE, -p.y * ORB_SCALE

def c(r, g, b, a=1.0):
    col = ColorRGBA(); col.r, col.g, col.b, col.a = r, g, b, a; return col

def make_tf(parent, child, tx, ty, tz, q, stamp):
    t = TransformStamped()
    t.header.stamp = stamp; t.header.frame_id = parent; t.child_frame_id = child
    t.transform.translation.x = tx
    t.transform.translation.y = ty
    t.transform.translation.z = tz
    t.transform.rotation = q
    return t


class TelloBridge:

    def __init__(self):
        rospy.init_node("tello_bridge", anonymous=False)

        self.pose_pub   = rospy.Publisher("/tello/pose",   PoseStamped, queue_size=1)
        self.path_pub   = rospy.Publisher("/tello/path",   Path,        queue_size=1)
        self.marker_pub = rospy.Publisher("/tello/marker", MarkerArray, queue_size=1)

        self._static_br  = tf2_ros.StaticTransformBroadcaster()
        self._dynamic_br = tf2_ros.TransformBroadcaster()
        self._pub_static_tf()

        self.path_msg = Path()
        self.path_msg.header.frame_id = FRAME_ID

        # 起飛點
        self._home_set = False
        self._home_rx = self._home_ry = 0.0

        # 回航狀態（來自 UDP 的 returning flag）
        self._returning = False

        # DR UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", UDP_PORT))
        self.sock.settimeout(1.0)

        # ORB correction UDP → Windows
        self._orb_sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._orb_target = ("127.0.0.1", ORB_CORR_PORT)

        # ========== ORB 狀態（改良版）==========
        self._orb_active      = False
        self._orb_last_t      = 0.0
        self._orb_hz_t        = 0.0
        self._orb_count       = 0
        self._orb_rx = self._orb_ry = self._orb_rz = 0.0
        self._orb_initialized = False
        self._orb_prev_lx = self._orb_prev_ly = 0.0   # 上一幀 ORB local 座標（算 delta 用）
        
        # 新增：ORB 失去追蹤時的處理
        self._orb_lost_since = None                    # 何時開始失去追蹤
        self._last_valid_orb_pos = (0.0, 0.0)          # 最後有效 ORB 位置
        self._last_valid_orb_time = 0.0                 # 最後有效時間戳
        
        # 新增：ORB 與 DR 的偏移量（用於恢復時對齊）
        self._orb_dr_offset_x = 0.0
        self._orb_dr_offset_y = 0.0
        self._offset_initialized = False
        
        self._dr_rx = self._dr_ry = 0.0  # 只需水平

        rospy.Subscriber("/orb_slam3/camera_pose", PoseStamped,
                         self._orb_pose_cb, queue_size=1)

        self._syscmd_pub = rospy.Publisher(
            "/orb_slam3_ros/syscommand", String, queue_size=1)
        threading.Thread(target=self._auto_reset_orb, daemon=True).start()
        rospy.loginfo("[bridge] Ready  DR:9999  ORB:/orb_slam3/camera_pose  corr:9997")

    def _pub_static_tf(self):
        now = rospy.Time.now()
        tfs = []
        for parent, child in [("map", "world"), ("world", "camera")]:
            st = TransformStamped()
            st.header.stamp = now; st.header.frame_id = parent
            st.child_frame_id = child; st.transform.rotation.w = 1.0
            tfs.append(st)
        self._static_br.sendTransform(tfs)

    def _auto_reset_orb(self):
        time.sleep(3.0)
        try:
            msg = String(); msg.data = "reset"
            self._syscmd_pub.publish(msg)
            rospy.loginfo("[bridge] ORB auto-reset sent")
        except Exception as e:
            rospy.logwarn(f"[bridge] auto-reset: {e}")

    def _orb_pose_cb(self, msg: PoseStamped):
        now_t = rospy.Time.now().to_sec()
        if now_t - self._orb_hz_t < 0.05:
            return
        self._orb_hz_t = now_t

        p = msg.pose.position
        lx, ly, lz = orb_to_local(p)

        gap = now_t - self._orb_last_t

        # ------------------------------------------------------------
        # 第一次初始化 ORB
        # ------------------------------------------------------------
        if not self._orb_initialized:
            # 以當前 DR 位置為基準
            self._orb_rx = self._dr_rx
            self._orb_ry = self._dr_ry
            self._orb_rz = 0.0
            self._orb_prev_lx = lx
            self._orb_prev_ly = ly
            self._orb_initialized = True
            self._orb_last_t = now_t
            self._orb_active = True
            self._orb_lost_since = None
            self._orb_count += 1
            
            # 記錄最後有效位置
            self._last_valid_orb_pos = (self._orb_rx, self._orb_ry)
            self._last_valid_orb_time = now_t
            
            # 初始化偏移量
            self._orb_dr_offset_x = self._dr_rx - self._orb_rx
            self._orb_dr_offset_y = self._dr_ry - self._orb_ry
            self._offset_initialized = True
            
            rospy.loginfo(f"[bridge] ORB session start at DR=({self._dr_rx:.2f},{self._dr_ry:.2f})")
            return

        # ------------------------------------------------------------
        # ORB 中斷後重新出現（改良版：不重置，繼續累積）
        # ------------------------------------------------------------
        if gap > ORB_TIMEOUT:
            # 更新 prev 座標，但**不重置位置**
            self._orb_prev_lx = lx
            self._orb_prev_ly = ly
            self._orb_last_t = now_t
            
            # 重要：不重置 _orb_rx/_orb_ry，繼續使用之前累積的位置
            lost_duration = now_t - self._orb_lost_since if self._orb_lost_since else 0
            
            if self._last_valid_orb_pos is not None:
                # 使用最後有效位置 + 偏移量
                self._orb_rx, self._orb_ry = self._last_valid_orb_pos
                rospy.loginfo(f"[bridge] ORB recovered after {lost_duration:.1f}s, "
                            f"continuing from ({self._orb_rx:.2f},{self._orb_ry:.2f})")
            else:
                rospy.loginfo(f"[bridge] ORB re-appeared after {gap:.1f}s, continuing from last pos")
            
            # 重置失去追蹤計時器
            self._orb_lost_since = None
            self._orb_active = True
            return

        # ------------------------------------------------------------
        # 正常情況：用相對位移（delta）累積，不依賴 ORB 絕對座標
        # ------------------------------------------------------------
        dlx = lx - self._orb_prev_lx
        dly = ly - self._orb_prev_ly
        
        # 檢查位移是否合理（避免跳躍）
        delta_dist = math.sqrt(dlx**2 + dly**2)
        if delta_dist < 100:  # 單幀位移小於 100cm 才接受（防異常跳躍）
            self._orb_rx += dlx
            self._orb_ry += dly
            
        self._orb_prev_lx = lx
        self._orb_prev_ly = ly

        self._orb_last_t = now_t
        self._orb_active = True
        self._orb_lost_since = None
        self._orb_count += 1
        
        # 記錄最後有效位置
        self._last_valid_orb_pos = (self._orb_rx, self._orb_ry)
        self._last_valid_orb_time = now_t
        
        if self._orb_count == 1:
            rospy.loginfo("[bridge] *** ORB ON ***")

        # 更新偏移量（用於 DR 輔助）
        if self._offset_initialized:
            # 平滑更新偏移量（EMA）
            alpha = 0.1
            new_offset_x = self._dr_rx - self._orb_rx
            new_offset_y = self._dr_ry - self._orb_ry
            self._orb_dr_offset_x = alpha * new_offset_x + (1 - alpha) * self._orb_dr_offset_x
            self._orb_dr_offset_y = alpha * new_offset_y + (1 - alpha) * self._orb_dr_offset_y

        # 發送修正後的 pose 給 Windows
        payload = json.dumps({
            "type": "orb_pose",
            "x": round(p.z * 100.0, 2),
            "y": round(-p.y * 100.0, 2),
            "z": round(-p.x * 100.0, 2),
            "tracking": True,
        }).encode()
        try:
            self._orb_sock.sendto(payload, self._orb_target)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[bridge] ORB send: {e}")

    def _publish(self, d):
        now = rospy.Time.now()

        # 起飛點（只設一次）
        if not self._home_set:
            self._home_rx, self._home_ry, _ = dr_to_rviz(d["home"][0], d["home"][1])
            self._home_set = True

        # 積分水平座標
        self._dr_rx, self._dr_ry, _ = dr_to_rviz(d["x"], d["z"])

        # 回航 flag
        self._returning = d.get("returning", False)

        # --------------------------------------------------------
        # 改良版：位置來源選擇
        # 1. ORB 有效 → 使用 ORB 位置
        # 2. ORB 失效但最近有有效位置 → 使用最後有效 ORB 位置 + DR 偏移量
        # 3. 完全失效 → 使用 DR
        # --------------------------------------------------------
        now_t = time.time()
        
        if self._orb_active and self._orb_initialized:
            # ORB 有效：使用 ORB 位置
            rx, ry, rz = self._orb_rx, self._orb_ry, 0.0
            source = "ORB"
            self._orb_lost_since = None
            
        elif self._last_valid_orb_pos is not None and (now_t - self._last_valid_orb_time) < ORB_MAX_LOST_TIME:
            # ORB 剛失去不久：使用最後有效 ORB 位置 + DR 的相對運動
            last_x, last_y = self._last_valid_orb_pos
            
            # 計算從最後有效時間到現在的 DR 位移
            dr_delta_x = self._dr_rx - (self._dr_rx_at_last_orb if hasattr(self, '_dr_rx_at_last_orb') else 0)
            dr_delta_y = self._dr_ry - (self._dr_ry_at_last_orb if hasattr(self, '_dr_ry_at_last_orb') else 0)
            
            rx = last_x + dr_delta_x
            ry = last_y + dr_delta_y
            rz = 0.0
            source = "ORB+DR"
            
            # 記錄失去追蹤時間
            if self._orb_lost_since is None:
                self._orb_lost_since = now_t
                self._dr_rx_at_last_orb = self._dr_rx
                self._dr_ry_at_last_orb = self._dr_ry
                
        else:
            # ORB 完全失效：使用 DR
            rx, ry, rz = self._dr_rx, self._dr_ry, 0.0
            source = "DR"
            self._orb_lost_since = None

        # 記錄最後有效 ORB 位置（用於下次失去時）
        if self._orb_active:
            self._last_valid_orb_pos = (rx, ry)
            self._last_valid_orb_time = now_t
            self._dr_rx_at_last_orb = self._dr_rx
            self._dr_ry_at_last_orb = self._dr_ry

        # TF
        q = yaw_to_quat(d["yaw"])
        self._dynamic_br.sendTransform(
            make_tf("world", "tello", rx, ry, rz, q, now))

        # Pose
        pose = PoseStamped()
        pose.header.stamp = now; pose.header.frame_id = FRAME_ID
        pose.pose.position.x = rx
        pose.pose.position.y = ry
        pose.pose.position.z = rz
        pose.pose.orientation = q
        self.pose_pub.publish(pose)

        # Path（純俯視軌跡，Z=0）
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > MAX_PATH:
            self.path_msg.poses = self.path_msg.poses[-MAX_PATH:]
        self.path_pub.publish(self.path_msg)

        # Markers
        ma = MarkerArray()
        if self._home_set:
            # 起飛點綠球
            hm = Marker()
            hm.header.stamp = now; hm.header.frame_id = FRAME_ID
            hm.ns, hm.id = "tello", 0
            hm.type = Marker.SPHERE; hm.action = Marker.ADD
            hm.pose.position.x = self._home_rx
            hm.pose.position.y = self._home_ry
            hm.pose.position.z = 0.0
            hm.pose.orientation.w = 1.0
            hm.scale.x = hm.scale.y = hm.scale.z = 0.12
            hm.color = c(0.0, 1.0, 0.2)
            ma.markers.append(hm)

            # 來源指示球（無人機正上方）
            sm = Marker()
            sm.header.stamp = now; sm.header.frame_id = FRAME_ID
            sm.ns, sm.id = "tello", 1
            sm.type = Marker.SPHERE; sm.action = Marker.ADD
            sm.pose.position.x = rx
            sm.pose.position.y = ry
            sm.pose.position.z = 0.15
            sm.pose.orientation.w = 1.0
            sm.scale.x = sm.scale.y = sm.scale.z = 0.08
            
            # 顏色：ORB=綠色，ORB+DR=黃色，DR=橘色
            if source == "ORB":
                sm.color = c(0.0, 1.0, 0.0)
            elif source == "ORB+DR":
                sm.color = c(1.0, 1.0, 0.0)  # 黃色
            else:
                sm.color = c(1.0, 0.5, 0.0)  # 橘色
            ma.markers.append(sm)

            # 顯示來源文字
            txt = Marker()
            txt.header.stamp = now; txt.header.frame_id = FRAME_ID
            txt.ns, txt.id = "tello", 7
            txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
            txt.pose.position.x = rx
            txt.pose.position.y = ry
            txt.pose.position.z = 0.3
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.15
            txt.color = c(1.0, 1.0, 1.0)
            txt.text = source
            ma.markers.append(txt)

            # 回航路徑：DJI 風格綠色三角錐
            if self._returning:
                # 實線：當前位置 → 起飛點
                line = Marker()
                line.header.stamp = now; line.header.frame_id = FRAME_ID
                line.ns, line.id = "tello", 2
                line.type = Marker.LINE_STRIP; line.action = Marker.ADD
                line.scale.x = 0.04
                line.color = c(0.0, 1.0, 0.1, 0.9)
                line.pose.orientation.w = 1.0
                line.points = [
                    Point(x=rx,               y=ry,               z=0.0),
                    Point(x=self._home_rx,    y=self._home_ry,    z=0.0),
                ]
                ma.markers.append(line)

                # 三角錐：左右各一條輔助線
                ddx = self._home_rx - rx
                ddy = self._home_ry - ry
                dist = math.sqrt(ddx**2 + ddy**2)
                cone_w = min(dist * 0.18, 0.6)

                if dist > 0.01:
                    perp_x = -ddy / dist * cone_w
                    perp_y =  ddx / dist * cone_w

                    for side, mid in [(-1, 3), (1, 4)]:
                        edge = Marker()
                        edge.header.stamp = now; edge.header.frame_id = FRAME_ID
                        edge.ns, edge.id = "tello", mid
                        edge.type = Marker.LINE_STRIP; edge.action = Marker.ADD
                        edge.scale.x = 0.02
                        edge.color = c(0.0, 0.9, 0.1, 0.5)
                        edge.pose.orientation.w = 1.0
                        edge.points = [
                            Point(x=rx + side * perp_x,
                                  y=ry + side * perp_y,
                                  z=0.0),
                            Point(x=self._home_rx, y=self._home_ry, z=0.0),
                        ]
                        ma.markers.append(edge)

                    # 三角底邊
                    base = Marker()
                    base.header.stamp = now; base.header.frame_id = FRAME_ID
                    base.ns, base.id = "tello", 5
                    base.type = Marker.LINE_STRIP; base.action = Marker.ADD
                    base.scale.x = 0.02
                    base.color = c(0.0, 0.9, 0.1, 0.4)
                    base.pose.orientation.w = 1.0
                    base.points = [
                        Point(x=rx - perp_x, y=ry - perp_y, z=0.0),
                        Point(x=rx + perp_x, y=ry + perp_y, z=0.0),
                    ]
                    ma.markers.append(base)

                    # 填充三角
                    tri = Marker()
                    tri.header.stamp = now; tri.header.frame_id = FRAME_ID
                    tri.ns, tri.id = "tello", 6
                    tri.type = Marker.TRIANGLE_LIST; tri.action = Marker.ADD
                    tri.scale.x = tri.scale.y = tri.scale.z = 1.0
                    tri.color = c(0.0, 1.0, 0.15, 0.22)
                    tri.pose.orientation.w = 1.0
                    apex = Point(x=self._home_rx, y=self._home_ry, z=0.0)
                    bl   = Point(x=rx - perp_x, y=ry - perp_y, z=0.0)
                    br   = Point(x=rx + perp_x, y=ry + perp_y, z=0.0)
                    tri.points = [apex, bl, br]
                    ma.markers.append(tri)

            else:
                # 回航結束：清除三角錐 marker
                for mid in [2, 3, 4, 5, 6, 7]:
                    clr = Marker()
                    clr.header.stamp = now; clr.header.frame_id = FRAME_ID
                    clr.ns, clr.id = "tello", mid
                    clr.action = Marker.DELETE
                    ma.markers.append(clr)

        self.marker_pub.publish(ma)
        rospy.loginfo_throttle(2.0,
            f"[{source}] ({rx:.2f},{ry:.2f})m  yaw={d['yaw']:.0f}°  "
            f"pts={len(self.path_msg.poses)}  RTH={self._returning}  "
            f"ORB_active={self._orb_active}")

    def run(self):
        last_data = None
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                raw, _ = self.sock.recvfrom(2048)
                last_data = json.loads(raw.decode())
            except socket.timeout:
                pass
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[bridge] recv: {e}")
            if last_data:
                try:
                    self._publish(last_data)
                except Exception as e:
                    rospy.logwarn(f"[bridge] publish: {e}")
            rate.sleep()

    def close(self):
        try: self.sock.close(); self._orb_sock.close()
        except: pass


if __name__ == "__main__":
    bridge = TelloBridge()
    try:
        bridge.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        bridge.close()