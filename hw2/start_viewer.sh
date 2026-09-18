#!/bin/bash
# start_viewer.sh
set -e

novnc_port=${NOVNC_PORT:-6080}
novnc_host_port=${NOVNC_HOST_PORT:-6080}
vnc_port=${VNC_PORT:-5900}
display_number=${DISPLAY_NUMBER:-1}


echo "Cleaning up any existing services..."
pkill -f "websockify.*${novnc_port}" 2>/dev/null || true
pkill -f "x11vnc.*${vnc_port}" 2>/dev/null || true
pkill -f "Xvfb.*:${display_number}" 2>/dev/null || true
sleep 1



Xvfb :${display_number} -screen 0 1280x800x24 &
sleep 1
export DISPLAY=:${display_number}
x11vnc -display :${display_number} -rfbport "${vnc_port}" -forever -nopw -quiet -shared &
websockify --web=/usr/share/novnc/ "${novnc_port}" localhost:"${vnc_port}" &


sleep 1

echo "✓ Services started"
echo "  Access noVNC at: http://localhost:${novnc_host_port}"

trap "kill $WEBSOCKIFY_PID $X11VNC_PID $XVFB_PID 2>/dev/null || true" EXIT INT TERM

python3 mhs_sim_test.py --viewer
