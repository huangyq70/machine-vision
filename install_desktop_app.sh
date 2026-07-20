#!/usr/bin/env bash
#
# install_desktop_app.sh -- install the Tanda Liquid Monitor as a desktop app
# with your logo as the icon.
#
# Usage:
#   ./install_desktop_app.sh                       # icon = ~/Pictures/tanda-logo.png
#   ./install_desktop_app.sh /path/to/logo.png     # custom icon path
#
# Creates a launcher in the applications menu AND on the Desktop.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ICON_SRC="${1:-$HOME/Pictures/tanda-logo.png}"
ICON_DST="$HOME/.local/share/icons/tanda-logo.png"
APP_DIR="$HOME/.local/share/applications"
DESKTOP_FILE="$APP_DIR/tanda-liquid-monitor.desktop"

mkdir -p "$HOME/.local/share/icons" "$APP_DIR"

# Copy the icon to a stable location (so moving the original won't break it).
if [ -f "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$ICON_DST"
    echo "Icon installed from: $ICON_SRC"
else
    echo "WARNING: icon not found at '$ICON_SRC'."
    echo "         Put your logo there, or pass the path: ./install_desktop_app.sh /path/to/logo.png"
    ICON_DST="$ICON_SRC"   # reference it directly; fix later if needed
fi

chmod +x "$REPO_DIR/tanda_launch.sh"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Tanda Liquid Monitor
Comment=Liquid level detection and pump output
Exec=$REPO_DIR/tanda_launch.sh
Icon=$ICON_DST
Terminal=true
Categories=Utility;Science;
EOF
chmod +x "$DESKTOP_FILE"

# Also drop a copy on the Desktop and mark it trusted (so it runs on click).
if [ -d "$HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$HOME/Desktop/tanda-liquid-monitor.desktop"
    chmod +x "$HOME/Desktop/tanda-liquid-monitor.desktop"
    gio set "$HOME/Desktop/tanda-liquid-monitor.desktop" metadata::trusted true 2>/dev/null || true
fi

update-desktop-database "$APP_DIR" 2>/dev/null || true

echo ""
echo "Installed 'Tanda Liquid Monitor'."
echo "  * Find it in the application menu (Utility/Science), or"
echo "  * double-click the icon on your Desktop."
echo "Edit tanda_launch.sh to change the pump port, tube capacity, etc."
