#!/usr/bin/env python3
"""
bridge_node.py — WSL1 Ubuntu 20.04 + ROS Noetic
Receives position data from Windows PyCharm main_fly7.py via UDP 127.0.0.1:9999
and republishes as ROS topics for rviz 3D visualization.

Startup order:
  Terminal 1: roscore
  Terminal 2: source /opt/ros/noetic/setup.bash && python3 bridge_node.py
  Terminal 3: source /opt/ros/noetic/setup.bash && rviz -d tello_rviz.rviz

WSL1 note: shares 127.0.0.1 with Windows, UDP works directly, no extra config needed.

Coordinate conversion:
  FlightTracker  X=right(cm)  Y=up(cm)  Z=forward(cm)  yaw=clockwise deg
  ROS world      X=forward(m) Y=left(m) Z=up(m)        yaw=counter-clockwise deg

Published topics:
  /tello/pose   geometry_msgs/PoseStamped   drone position + orientation
  /tello/path   nav_msgs/Path               3D flight path
  /tello/marker visualization_msgs/MarkerArray
                  id=0  green sphere  home point
                  id=1  orange arrow  drone -> home direction
                  id=2  colored axes  drone body XYZ axes
"""

import socket, json, math, struct
import rospy
import tf2_ros
from geometry_msgs.msg import (PoseStamped, Point, TransformStamped,
                                Vector3, Quaternion)
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header

UDP_PORT     = 9999   # Windows → WSL1 位置
ORB_CORR_PORT = 9997  # WSL1 → Windows ORB 校正位置
FRAME_ID = "world"
SCALE    = 1      # cm → m
MAX_PATH = 8000


# -- Utility functions --
def yaw_to_quat(yaw_deg: float) -> Quaternion:
    """Tello clockwise yaw (deg) -> ROS quaternion (counter-clockwise around Z)"""
    half = math.radians(-yaw_deg) / 2.0
    q = Quaternion()
    q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(half), math.cos(half)
    return q


def to_rviz(x_cm, y_cm, z_cm):
    """FlightTracker cm → rviz world frame m"""
    return z_cm * SCALE, -x_cm * SCALE, y_cm * SCALE  # rx, ry, rz


def color(r, g, b, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


def make_marker(ns, mid, mtype, frame=FRAME_ID) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.id        = ns, mid
    m.type            = mtype
    m.action          = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


# -- Main class --
class TelloBridge:

    def __init__(self):
        rospy.init_node("tello_bridge", anonymous=False)

        # -- Publishers --
        self.pose_pub   = rospy.Publisher("/tello/pose",   PoseStamped,  queue_size=1)
        self.path_pub   = rospy.Publisher("/tello/path",   Path,         queue_size=1)
        self.marker_pub = rospy.Publisher("/tello/marker", MarkerArray,  queue_size=1)

        # -- Static TF: declare world frame so rviz Fixed Frame can resolve --
        self._static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._dynamic_broadcaster = tf2_ros.TransformBroadcaster()
        self._publish_world_frame()

        # -- Internal state --
        self.path_msg = Path()
        self.path_msg.header.frame_id = FRAME_ID
        self._home_set = False
        self._home_rx  = 0.0
        self._home_ry  = 0.0

        # -- UDP socket --
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", UDP_PORT))
        self.sock.settimeout(1.0)
        rospy.loginfo(f"[bridge] Listening UDP 127.0.0.1:{UDP_PORT}")

        # ORB-SLAM3 correction: subscribe pose, send back to Windows
        self._orb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._orb_target = ("127.0.0.1", ORB_CORR_PORT)
        self._orb_active = False   # True once ORB starts tracking
        self._orb_last_t = 0.0
        rospy.Subscriber("/orb_slam3/camera_pose", PoseStamped,
                          self._orb_pose_cb, queue_size=1)
        rospy.loginfo("[bridge] Subscribed /orb_slam3/camera_pose")

    def _publish_world_frame(self):
        """Publish static TFs: map->world and world->camera (identity)"""
        now = rospy.Time.now()

        # map -> world
        st1 = TransformStamped()
        st1.header.stamp    = now
        st1.header.frame_id = "map"
        st1.child_frame_id  = "world"
        st1.transform.rotation.w = 1.0

        # world -> camera (identity - ORB-SLAM3 needs this to exist)
        st2 = TransformStamped()
        st2.header.stamp    = now
        st2.header.frame_id = "world"
        st2.child_frame_id  = "camera"
        st2.transform.rotation.w = 1.0

        self._static_broadcaster.sendTransform([st1, st2])

    def _broadcast_drone_tf(self, rx, ry, rz, yaw_deg, now):
        """Publish dynamic TF world -> tello for advanced use"""
        t = TransformStamped()
        t.header.stamp    = now
        t.header.frame_id = "world"
        t.child_frame_id  = "tello"
        t.transform.translation.x = rx
        t.transform.translation.y = ry
        t.transform.translation.z = rz
        t.transform.rotation = yaw_to_quat(yaw_deg)
        self._dynamic_broadcaster.sendTransform(t)

    def _update_home(self, hx_cm, hz_cm):
        if not self._home_set:
            self._home_rx, self._home_ry, _ = to_rviz(hx_cm, 0, hz_cm)
            self._home_set = True
            rospy.loginfo(f"[bridge] Home point fixed rviz=({self._home_rx:.2f}, {self._home_ry:.2f})")

    def _publish(self, d):
        now = rospy.Time.now()
        rx, ry, rz = to_rviz(d["x"], d["y"], d["z"])
        self._update_home(d["home"][0], d["home"][1])

        # -- TF --
        self._broadcast_drone_tf(rx, ry, rz, d["yaw"], now)

        # -- PoseStamped --
        pose = PoseStamped()
        pose.header.stamp    = now
        pose.header.frame_id = FRAME_ID
        pose.pose.position.x = rx
        pose.pose.position.y = ry
        pose.pose.position.z = rz
        pose.pose.orientation = yaw_to_quat(d["yaw"])
        self.pose_pub.publish(pose)

        # -- Path 3D --
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > MAX_PATH:
            self.path_msg.poses = self.path_msg.poses[-MAX_PATH:]
        self.path_pub.publish(self.path_msg)

        # -- MarkerArray --
        ma = MarkerArray()

        # id=0 home point green sphere
        if self._home_set:
            home_m = make_marker("tello", 0, Marker.SPHERE)
            home_m.header.stamp    = now
            home_m.pose.position.x = self._home_rx
            home_m.pose.position.y = self._home_ry
            home_m.pose.position.z = 0.0
            home_m.scale.x = home_m.scale.y = home_m.scale.z = 0.12
            home_m.color = color(0.0, 1.0, 0.2)
            ma.markers.append(home_m)

            # id=1 return arrow (drone -> home)
            arrow = make_marker("tello", 1, Marker.ARROW)
            arrow.header.stamp = now
            arrow.points = [
                Point(x=rx,             y=ry,             z=rz),
                Point(x=self._home_rx,  y=self._home_ry,  z=0.0),
            ]
            arrow.scale.x = 0.04
            arrow.scale.y = 0.10
            arrow.scale.z = 0.0
            arrow.color = color(1.0, 0.5, 0.0, 0.9)
            ma.markers.append(arrow)

        # drone body axes removed - TF display handles this

        self.marker_pub.publish(ma)

        # Terminal log (every 2 seconds)
        if self._home_set:
            dist = math.sqrt((rx-self._home_rx)**2 + (ry-self._home_ry)**2 + rz**2)
            rospy.loginfo_throttle(2.0,
                f"pos=({rx:.2f},{ry:.2f},{rz:.2f})m "
                f"yaw={d['yaw']:.1f}deg dist_home={dist:.2f}m "
                f"path={len(self.path_msg.poses)}pts")

    def _orb_pose_cb(self, msg: PoseStamped):
        """
        Called when ORB-SLAM3 publishes a new pose.
        Converts ORB world frame → FlightTracker frame and sends UDP correction.

        ORB-SLAM3 (Monocular) coordinate system:
          X = right, Y = down, Z = forward  (camera optical frame)
        FlightTracker frame:
          X = right(cm), Y = up(cm), Z = forward(cm)
        Scale: ORB monocular is scale-ambiguous.
          We use get_height() from Tello as vertical reference to estimate scale.
          For now we send normalised ORB pose; main_fly7 applies scale.
        """
        import time as _time
        now = _time.time()
        if now - self._orb_last_t < 0.05:   # max 20 Hz to Windows
            return
        self._orb_last_t = now

        p = msg.pose.position
        # ORB camera frame → FlightTracker (cm)
        # ORB X→right, Y→down, Z→forward
        # FlightTracker X→right, Y→up, Z→forward
        orb_x =  p.x * 100.0   # right (cm)
        orb_y = -p.y * 100.0   # up    (cm) — flip Y
        orb_z =  p.z * 100.0   # forward (cm)

        self._orb_active = True
        payload = json.dumps({
            "type": "orb_pose",
            "x": round(orb_x, 2),
            "y": round(orb_y, 2),
            "z": round(orb_z, 2),
            "tracking": True,
        }).encode()
        try:
            self._orb_sock.sendto(payload, self._orb_target)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[bridge] ORB send error: {e}")

    def run(self):
        last_data = None
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                raw, _ = self.sock.recvfrom(2048)
                last_data = json.loads(raw.decode())
            except socket.timeout:
                pass
            except json.JSONDecodeError as e:
                rospy.logwarn(f"[bridge] JSON parse error: {e}")
            if last_data:
                try:
                    self._publish(last_data)
                except Exception as e:
                    rospy.logwarn(f"[bridge] Publish error: {e}")
            rate.sleep()

    def close(self):
        self.sock.close()
        self._orb_sock.close()
        rospy.loginfo("[bridge] Shutdown")


if __name__ == "__main__":
    bridge = TelloBridge()
    try:
        bridge.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        bridge.close()
