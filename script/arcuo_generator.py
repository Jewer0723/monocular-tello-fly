import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(aruco_dict, 0, 400)
cv2.imwrite("aruco_marker_id0.png", marker)