#!/usr/bin/env python3
"""
tello_cam_node.py  -  WSL1 Ubuntu 20.04 + ROS Noetic

Receives single-packet JPEG frames from Windows main_fly8.py via UDP :9998,
publishes as sensor_msgs/Image on /camera/image_raw for ORB-SLAM3.

Frame resolution: 480x360 (resized in main_fly8 to fit in one UDP packet)
Scaled intrinsics from tello.yaml (original 960x720, scale=0.5):
  fx=455.036  fy=457.047  cx=242.762  cy=168.359

Startup:
  source /opt/ros/noetic/setup.bash && python3 tello_cam_node.py
"""

import socket
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge

UDP_PORT = 9998
FRAME_W  = 480
FRAME_H  = 360

# Scaled intrinsics (original 960x720 * 0.5)
FX, FY = 455.036, 457.047
CX, CY = 242.762, 168.359
K1, K2 = -0.005941, 0.055161
P1, P2 = -0.006094, 0.003192


class TelloCamNode:

    def __init__(self):
        rospy.init_node("tello_cam_node", anonymous=False)
        self.bridge   = CvBridge()
        self.img_pub  = rospy.Publisher("/camera/image_raw",   Image,      queue_size=1)
        self.info_pub = rospy.Publisher("/camera/camera_info", CameraInfo, queue_size=1)
        self._build_camera_info()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", UDP_PORT))
        self.sock.settimeout(1.0)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        rospy.loginfo(f"[cam_node] Listening UDP 127.0.0.1:{UDP_PORT}  ({FRAME_W}x{FRAME_H})")

    def _build_camera_info(self):
        ci = CameraInfo()
        ci.width  = FRAME_W
        ci.height = FRAME_H
        ci.distortion_model = "plumb_bob"
        ci.K = [FX, 0,  CX,
                0,  FY, CY,
                0,  0,  1]
        ci.D = [K1, K2, P1, P2, 0.0]
        ci.R = [1, 0, 0,
                0, 1, 0,
                0, 0, 1]
        ci.P = [FX, 0,  CX, 0,
                0,  FY, CY, 0,
                0,  0,  1,  0]
        self._cam_info = ci

    def _publish(self, jpeg_bytes):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            rospy.logwarn_throttle(5.0, "[cam_node] Failed to decode JPEG")
            return

        now = rospy.Time.now()
        hdr = Header()
        hdr.stamp    = now
        hdr.frame_id = "camera"

        ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        ros_img.header = hdr
        self.img_pub.publish(ros_img)

        self._cam_info.header = hdr
        self.info_pub.publish(self._cam_info)

    def run(self):
        frame_count = 0
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            try:
                # Each UDP packet is one complete JPEG (< 65KB after resize)
                data, _ = self.sock.recvfrom(65535)
                # Verify it looks like a JPEG (SOI marker)
                if data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9':
                    self._publish(data)
                    frame_count += 1
                    if frame_count % 30 == 0:
                        rospy.loginfo(f"[cam_node] {frame_count} frames received")
                else:
                    rospy.logwarn_throttle(5.0,
                        f"[cam_node] Bad packet: {len(data)}B, "
                        f"header={data[:2].hex()} tail={data[-2:].hex()}")
            except socket.timeout:
                rospy.loginfo_throttle(5.0, "[cam_node] Waiting for frames from Windows...")
            except Exception as e:
                rospy.logwarn(f"[cam_node] Error: {e}")
            rate.sleep()

    def close(self):
        self.sock.close()
        rospy.loginfo("[cam_node] Shutdown")


if __name__ == "__main__":
    node = TelloCamNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.close()
