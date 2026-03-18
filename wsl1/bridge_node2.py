#!/usr/bin/env python3
"""
bridge_node2.py — WSL1 Ubuntu 20.04 + ROS Noetic  (DR only, ORB removed)

顯示：
  - 飛行軌跡（俯視，Z=0 固定）
  - 回航時：DJI 風格綠色三角錐
  - 起飛點綠球
  - DR 來源指示球
"""
import socket, json, math, threading, time
import rospy
import tf2_ros
from geometry_msgs.msg import Point, TransformStamped, Quaternion, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

UDP_PORT  = 9999
FRAME_ID  = "world"
SCALE     = 0.25
MAX_PATH  = 8000


def yaw_to_quat(yaw_deg):
    half = math.radians(-yaw_deg) / 2.0
    q = Quaternion()
    q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(half), math.cos(half)
    return q

def dr_to_rviz(x_cm, z_cm):
    return z_cm * SCALE, -x_cm * SCALE, 0.0

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

        self._home_set = False
        self._home_rx = self._home_ry = 0.0
        self._returning = False
        self._dr_rx = self._dr_ry = 0.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", UDP_PORT))
        self.sock.settimeout(1.0)

        rospy.loginfo("[bridge] Ready  DR:9999")

    def _pub_static_tf(self):
        now = rospy.Time.now()
        tfs = []
        for parent, child in [("map", "world"), ("world", "camera")]:
            st = TransformStamped()
            st.header.stamp = now; st.header.frame_id = parent
            st.child_frame_id = child; st.transform.rotation.w = 1.0
            tfs.append(st)
        self._static_br.sendTransform(tfs)

    def _publish(self, d):
        now = rospy.Time.now()

        if not self._home_set:
            self._home_rx, self._home_ry, _ = dr_to_rviz(d["home"][0], d["home"][1])
            self._home_set = True

        self._dr_rx, self._dr_ry, _ = dr_to_rviz(d["x"], d["z"])
        self._returning = d.get("returning", False)

        rx, ry, rz = self._dr_rx, self._dr_ry, 0.0

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

        # Path
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

            # DR 來源指示球
            sm = Marker()
            sm.header.stamp = now; sm.header.frame_id = FRAME_ID
            sm.ns, sm.id = "tello", 1
            sm.type = Marker.SPHERE; sm.action = Marker.ADD
            sm.pose.position.x = rx
            sm.pose.position.y = ry
            sm.pose.position.z = 0.15
            sm.pose.orientation.w = 1.0
            sm.scale.x = sm.scale.y = sm.scale.z = 0.08
            sm.color = c(1.0, 0.5, 0.0)   # 橘色 = DR
            ma.markers.append(sm)

            # 回航三角錐
            if self._returning:
                line = Marker()
                line.header.stamp = now; line.header.frame_id = FRAME_ID
                line.ns, line.id = "tello", 2
                line.type = Marker.LINE_STRIP; line.action = Marker.ADD
                line.scale.x = 0.04
                line.color = c(0.0, 1.0, 0.1, 0.9)
                line.pose.orientation.w = 1.0
                line.points = [
                    Point(x=rx,            y=ry,            z=0.0),
                    Point(x=self._home_rx, y=self._home_ry, z=0.0),
                ]
                ma.markers.append(line)

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
                            Point(x=rx + side*perp_x, y=ry + side*perp_y, z=0.0),
                            Point(x=self._home_rx,    y=self._home_ry,    z=0.0),
                        ]
                        ma.markers.append(edge)

                    tri = Marker()
                    tri.header.stamp = now; tri.header.frame_id = FRAME_ID
                    tri.ns, tri.id = "tello", 6
                    tri.type = Marker.TRIANGLE_LIST; tri.action = Marker.ADD
                    tri.scale.x = tri.scale.y = tri.scale.z = 1.0
                    tri.color = c(0.0, 1.0, 0.15, 0.22)
                    tri.pose.orientation.w = 1.0
                    apex = Point(x=self._home_rx, y=self._home_ry, z=0.0)
                    bl   = Point(x=rx-perp_x, y=ry-perp_y, z=0.0)
                    br   = Point(x=rx+perp_x, y=ry+perp_y, z=0.0)
                    tri.points = [apex, bl, br]
                    ma.markers.append(tri)

            else:
                for mid in [2, 3, 4, 5, 6]:
                    clr = Marker()
                    clr.header.stamp = now; clr.header.frame_id = FRAME_ID
                    clr.ns, clr.id = "tello", mid
                    clr.action = Marker.DELETE
                    ma.markers.append(clr)

        self.marker_pub.publish(ma)
        rospy.loginfo_throttle(2.0,
            f"[DR] ({rx:.2f},{ry:.2f})m  yaw={d['yaw']:.0f}°  "
            f"pts={len(self.path_msg.poses)}  RTH={self._returning}")

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
        try: self.sock.close()
        except: pass


if __name__ == "__main__":
    bridge = TelloBridge()
    try:
        bridge.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        bridge.close()
