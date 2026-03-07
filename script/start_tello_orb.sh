#!/bin/bash
# =============================================================================
# start_tello_orb.sh  -  Tello ORB-SLAM3 + RViz pipeline (WSL1 Ubuntu 20.04)
# =============================================================================

CATKIN_WS="$HOME/catkin_build"
ORB_VOC="$HOME/ORB_SLAM3/Vocabulary/ORBvoc.txt"
ORB_SETTINGS="$CATKIN_WS/src/orb_slam3_ros_wrapper/config/tello.yaml"
LAUNCH_FILE="orb_slam3_mono_tello.launch"
LAUNCH_PATH="$CATKIN_WS/src/orb_slam3_ros_wrapper/launch/$LAUNCH_FILE"
BRIDGE_DIR="$HOME/tello_bridge"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}   Tello ORB-SLAM3 Bridge Startup${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

# ── Pre-flight checks ──────────────────────────────────────────────────────
ERRORS=0
ok()   { echo -e "${GREEN}[OK]     $1${NC}"; }
fail() { echo -e "${RED}[MISS]   $1${NC}"; ERRORS=$((ERRORS+1)); }

echo -e "${YELLOW}--- Checking files ---${NC}"
[ -f "$ORB_VOC" ]                         && ok "$ORB_VOC"          || fail "$ORB_VOC"
[ -f "$ORB_SETTINGS" ]                    && ok "$ORB_SETTINGS"     || fail "$ORB_SETTINGS"
[ -f "$LAUNCH_PATH" ]                     && ok "$LAUNCH_PATH"      || fail "$LAUNCH_PATH"
[ -f "$BRIDGE_DIR/tello_cam_node.py" ]    && ok "tello_cam_node.py" || fail "$BRIDGE_DIR/tello_cam_node.py"
[ -f "$BRIDGE_DIR/bridge_node.py" ]       && ok "bridge_node.py"    || fail "$BRIDGE_DIR/bridge_node.py"
[ -f "$BRIDGE_DIR/tello_rviz.rviz" ]      && ok "tello_rviz.rviz"  || fail "$BRIDGE_DIR/tello_rviz.rviz"
[ -d "$CATKIN_WS/devel" ]                 && ok "catkin devel/"     || fail "$CATKIN_WS/devel"

# Check hector_trajectory_server
if rospack find hector_trajectory_server &>/dev/null; then
    ok "hector_trajectory_server"
else
    echo -e "${YELLOW}[WARN]   hector_trajectory_server not found - install with:${NC}"
    echo -e "         sudo apt install ros-noetic-hector-trajectory-server"
fi

echo ""
[ $ERRORS -gt 0 ] && { echo -e "${RED}$ERRORS missing. Fix then retry.${NC}"; exit 1; }

source /opt/ros/noetic/setup.bash
source "$CATKIN_WS/devel/setup.bash"
export DISPLAY=:0 LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1 OGRE_RTT_MODE=Copy

# ── Launch nodes ───────────────────────────────────────────────────────────
run_xterm() {
    local TITLE="$1"; local CMD="$2"
    xterm -T "$TITLE" -fa 'Monospace' -fs 11 -e bash -c "
        source /opt/ros/noetic/setup.bash
        source $CATKIN_WS/devel/setup.bash
        export DISPLAY=:0 LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1 OGRE_RTT_MODE=Copy
        echo '=== $TITLE ==='
        $CMD
        echo '[Exited] Press Enter'; read
    " &
    sleep 0.8
}

echo -e "${YELLOW}--- Starting nodes ---${NC}"

echo -e "${CYAN}[1/5] roscore${NC}"
run_xterm "roscore" "roscore"
sleep 2

echo -e "${CYAN}[2/5] tello_cam_node  (UDP :9998 -> /camera/image_raw)${NC}"
run_xterm "Tello Camera Node" "cd $BRIDGE_DIR && python3 tello_cam_node.py"
sleep 1

echo -e "${CYAN}[3/5] ORB-SLAM3 Mono Tello${NC}"
run_xterm "ORB-SLAM3" "roslaunch orb_slam3_ros_wrapper $LAUNCH_FILE"
sleep 2

echo -e "${CYAN}[4/5] bridge_node  (pos bridge + ORB correction)${NC}"
run_xterm "Bridge Node" "cd $BRIDGE_DIR && python3 bridge_node.py"
sleep 1

echo -e "${CYAN}[5/5] rviz${NC}"
run_xterm "RViz" "rviz -d $BRIDGE_DIR/tello_rviz.rviz"

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}  All nodes started!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo -e "Start ${YELLOW}main_fly8.py${NC} on Windows."
echo -e "Move drone slowly after takeoff to initialize ORB-SLAM3."
echo -e "HUD: ${YELLOW}ORB: INIT${NC} -> ${GREEN}ORB: ON${NC} when correction active."
echo ""
