#!/usr/bin/env python3
"""
bridge_node.py — WSL1 Ubuntu 20.04 + ROS Noetic

位置來源：
  ORB 有輸出 → 使用 ORB 座標（精確）
  ORB lost   → fallback 積分座標（DR）

ORB 問題修正：
  - 第一幀 ORB 出現時，從當前積分位置開始（不從原點跳）
  - ORB 再次 lost 後恢復，同樣從當前積分位置接續
  - ORB_DISPLAY_SCALE 放大 ORB 座標顯示

無人機上方小球：綠=ORB 橘=DR
"""
import socket, json, math, threading, time
import rospy
import tf2_ros
from geometry_msgs.msg import (PoseStamped, Point, TransformStamped, Quaternion)
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String

UDP_PORT          = 9999
ORB_CORR_PORT     = 9997
FRAME_ID          = "world"
SCALE             = 1     # DR cm → m
ORB_DISPLAY_SCALE = 3.0      # ORB 座標放大倍數（調整軌跡大小）
MAX_PATH          = 8000
ORB_TIMEOUT       = 1.5      # 超過幾秒無 ORB → fallback DR


def yaw_to_quat(yaw_deg):
    half = math.radians(-yaw_deg) / 2.0
    q = Quaternion()
    q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(half), math.cos(half)
    return q

def dr_to_rviz(x_cm, y_cm, z_cm):
    return z_cm * SCALE, -x_cm * SCALE, y_cm * SCALE

def orb_to_local(p, scale):
    """ORB optical → local offset (m, 放大)"""
    return p.z * scale, -p.x * scale, -p.y * scale

def color(r, g, b, a=1.0):
    c = ColorRGBA(); c.r, c.g, c.b, c.a = r, g, b, a; return c

def make_marker(ns, mid, mtype):
    m = Marker(); m.header.frame_id = FRAME_ID
    m.ns, m.id = ns, mid; m.type = mtype
    m.action = Marker.ADD; m.pose.orientation.w = 1.0; return m


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

        self._home_set = False
        self._home_rx = self._home_ry = 0.0

        # DR UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", UDP_PORT))
        self.sock.settimeout(1.0)

        # ORB corr UDP (→ Windows)
        self._orb_sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._orb_target = ("127.0.0.1", ORB_CORR_PORT)

        # ORB 狀態
        self._orb_active  = False
        self._orb_last_t  = 0.0
        self._orb_hz_t    = 0.0
        self._orb_count   = 0

        # ORB 當前絕對座標（= ORB local + offset）
        self._orb_rx = self._orb_ry = self._orb_rz = 0.0

        # offset：讓 ORB 軌跡從積分座標接續
        # 每次 ORB 從無到有時重新計算
        self._orb_ox = self._orb_oy = self._orb_oz = 0.0
        self._orb_initialized = False  # 本次 ORB session 是否已設 offset

        # 當前積分座標（從 DR UDP 更新，供 ORB offset 計算用）
        self._dr_rx = self._dr_ry = self._dr_rz = 0.0

        rospy.Subscriber("/orb_slam3/camera_pose", PoseStamped,
                         self._orb_pose_cb, queue_size=1)
        rospy.loginfo("[bridge] Ready. DR:9999  ORB-sub:/orb_slam3/camera_pose  corr:9997")

        self._syscmd_pub = rospy.Publisher(
            "/orb_slam3_ros/syscommand", String, queue_size=1)
        threading.Thread(target=self._auto_reset_orb, daemon=True).start()

    def _pub_static_tf(self):
        now = rospy.Time.now()
        tfs = []
        for parent, child in [("map","world"),("world","camera")]:
            st = TransformStamped()
            st.header.stamp = now
            st.header.frame_id = parent
            st.child_frame_id  = child
            st.transform.rotation.w = 1.0
            tfs.append(st)
        self._static_br.sendTransform(tfs)

    def _broadcast_tf(self, rx, ry, rz, yaw_deg, now):
        t = TransformStamped()
        t.header.stamp = now; t.header.frame_id = "world"; t.child_frame_id = "tello"
        t.transform.translation.x = rx
        t.transform.translation.y = ry
        t.transform.translation.z = rz
        t.transform.rotation = yaw_to_quat(yaw_deg)
        self._dynamic_br.sendTransform(t)

    def _update_home(self, hx_cm, hz_cm):
        if not self._home_set:
            self._home_rx, self._home_ry, _ = dr_to_rviz(hx_cm, 0, hz_cm)
            self._home_set = True
            rospy.loginfo(f"[bridge] Home=({self._home_rx:.2f},{self._home_ry:.2f})")

    def _orb_pose_cb(self, msg: PoseStamped):
        now_t = rospy.Time.now().to_sec()
        if now_t - self._orb_hz_t < 0.05:
            return
        self._orb_hz_t = now_t

        p = msg.pose.position
        # ORB local offset (含放大)
        lx, ly, lz = orb_to_local(p, ORB_DISPLAY_SCALE)

        # 第一幀或 ORB 重新出現時：計算 offset 讓軌跡從積分位置接續
        if not self._orb_initialized:
            self._orb_ox = self._dr_rx - lx
            self._orb_oy = self._dr_ry - ly
            self._orb_oz = self._dr_rz - lz
            self._orb_initialized = True
            rospy.loginfo(
                f"[bridge] ORB session start, offset="
                f"({self._orb_ox:.2f},{self._orb_oy:.2f},{self._orb_oz:.2f})")

        self._orb_rx = lx + self._orb_ox
        self._orb_ry = ly + self._orb_oy
        self._orb_rz = lz + self._orb_oz

        self._orb_last_t = now_t
        self._orb_active  = True
        self._orb_count  += 1

        if self._orb_count == 1:
            rospy.loginfo("[bridge] *** ORB ON ***")
        if self._orb_count % 100 == 0:
            rospy.loginfo(
                f"[bridge] ORB frames={self._orb_count} "
                f"pos=({self._orb_rx:.2f},{self._orb_ry:.2f},{self._orb_rz:.2f})m")

        # 回傳 Windows (cm)
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
            rospy.logwarn_throttle(5.0, f"[bridge] ORB send err: {e}")

    def _auto_reset_orb(self):
        time.sleep(3.0)
        try:
            msg = String(); msg.data = "reset"
            self._syscmd_pub.publish(msg)
            rospy.loginfo("[bridge] Auto-reset sent")
        except Exception as e:
            rospy.logwarn(f"[bridge] Auto-reset failed: {e}")

    def _publish(self, d):
        now = rospy.Time.now()
        self._update_home(d["home"][0], d["home"][1])

        # 更新積分座標（永遠更新，供 ORB offset 計算）
        self._dr_rx, self._dr_ry, self._dr_rz = dr_to_rviz(d["x"], d["y"], d["z"])

        # 位置來源選擇
        orb_alive = self._orb_active and \
                    (now.to_sec() - self._orb_last_t) < ORB_TIMEOUT

        if orb_alive:
            rx, ry, rz = self._orb_rx, self._orb_ry, self._orb_rz
            source = "ORB"
        else:
            # ORB lost：重置 _orb_initialized 讓下次 ORB 出現時重新對齊
            if self._orb_initialized:
                self._orb_initialized = False
                rospy.loginfo("[bridge] ORB lost → DR fallback, will re-align on next ORB")
            rx, ry, rz = self._dr_rx, self._dr_ry, self._dr_rz
            source = "DR"

        # TF
        self._broadcast_tf(rx, ry, rz, d["yaw"], now)

        # Pose
        pose = PoseStamped()
        pose.header.stamp = now; pose.header.frame_id = FRAME_ID
        pose.pose.position.x = rx
        pose.pose.position.y = ry
        pose.pose.position.z = rz
        pose.pose.orientation = yaw_to_quat(d["yaw"])
        self.pose_pub.publish(pose)

        # Path
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > MAX_PATH:
            self.path_msg.poses = self.path_msg.poses[-MAX_PATH:]
        self.path_pub.publish(self.path_msg)

        # Markers
        ma = MarkerArray()
        if self._home_set:
            # 起飛點
            hm = make_marker("tello", 0, Marker.SPHERE)
            hm.header.stamp = now
            hm.pose.position.x = self._home_rx
            hm.pose.position.y = self._home_ry
            hm.pose.position.z = 0.0
            hm.scale.x = hm.scale.y = hm.scale.z = 0.12
            hm.color = color(0.0, 1.0, 0.2)
            ma.markers.append(hm)

            # 回家箭頭
            arr = make_marker("tello", 1, Marker.ARROW)
            arr.header.stamp = now
            arr.points = [
                Point(x=rx, y=ry, z=rz),
                Point(x=self._home_rx, y=self._home_ry, z=0.0),
            ]
            arr.scale.x = 0.04; arr.scale.y = 0.10; arr.scale.z = 0.0
            arr.color = color(1.0, 0.5, 0.0, 0.9)
            ma.markers.append(arr)

            # 來源指示球（無人機正上方）
            sm = make_marker("tello", 2, Marker.SPHERE)
            sm.header.stamp = now
            sm.pose.position.x = rx
            sm.pose.position.y = ry
            sm.pose.position.z = rz + 0.25
            sm.scale.x = sm.scale.y = sm.scale.z = 0.10
            sm.color = color(0.0, 1.0, 0.0) if source == "ORB" else color(1.0, 0.5, 0.0)
            ma.markers.append(sm)

        self.marker_pub.publish(ma)

        if self._home_set:
            dist = math.sqrt((rx-self._home_rx)**2 + (ry-self._home_ry)**2 + rz**2)
            rospy.loginfo_throttle(2.0,
                f"[{source}] ({rx:.2f},{ry:.2f},{rz:.2f})m "
                f"yaw={d['yaw']:.0f}° dist={dist:.2f}m pts={len(self.path_msg.poses)}")

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
                rospy.logwarn_throttle(5.0, f"[bridge] recv err: {e}")
            if last_data:
                try:
                    self._publish(last_data)
                except Exception as e:
                    rospy.logwarn(f"[bridge] publish err: {e}")
            rate.sleep()

    def close(self):
        try: self.sock.close(); self._orb_sock.close()
        except: pass
        rospy.loginfo("[bridge] Shutdown")


if __name__ == "__main__":
    bridge = TelloBridge()
    try:
        bridge.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        bridge.close()
