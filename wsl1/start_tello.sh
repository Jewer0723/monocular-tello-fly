#!/bin/bash
# =============================================================================
# start_tello.sh  -  Tello RViz bridge pipeline (WSL1 Ubuntu 20.04)
#                    ORB-SLAM3 removed; DR-only bridge_node2.py
# =============================================================================

CATKIN_WS="$HOME/catkin_build"
BRIDGE_DIR="$HOME/tello_bridge"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}   Tello RViz Bridge Startup${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

# ── Pre-flight checks ──────────────────────────────────────────────────────
ERRORS=0
ok()   { echo -e "${GREEN}[OK]     $1${NC}"; }
fail() { echo -e "${RED}[MISS]   $1${NC}"; ERRORS=$((ERRORS+1)); }

echo -e "${YELLOW}--- Checking files ---${NC}"
[ -f "$BRIDGE_DIR/bridge_node2.py" ]       && ok "bridge_node2.py"   || fail "$BRIDGE_DIR/bridge_node2.py"
[ -f "$BRIDGE_DIR/tello_rviz2.rviz" ]      && ok "tello_rviz2.rviz"  || fail "$BRIDGE_DIR/tello_rviz2.rviz"
[ -d "$CATKIN_WS/devel" ]                  && ok "catkin devel/"      || fail "$CATKIN_WS/devel"

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

echo -e "${CYAN}[1/3] roscore${NC}"
run_xterm "roscore" "roscore"
sleep 2

echo -e "${CYAN}[2/3] bridge_node2  (DR pos bridge UDP:9999 → RViz)${NC}"
run_xterm "Bridge Node" "cd $BRIDGE_DIR && python3 bridge_node2.py"
sleep 1

echo -e "${CYAN}[3/3] rviz${NC}"
run_xterm "RViz" "rviz -d $BRIDGE_DIR/tello_rviz2.rviz"

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}  All nodes started!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo -e "Start ${YELLOW}main_fly9.1.py${NC} on Windows."
echo ""
