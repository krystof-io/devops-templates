#!/bin/sh
set -e

echo "Downloading FFmpeg based on versions.yaml..."

FFMPEG_BRANCH=$(grep -A 10 "^ffmpeg:" /tmp/versions.yaml | grep "branch:" | sed 's/.*"\(.*\)".*/\1/')
FFMPEG_AUTOBUILD=$(grep -A 10 "^ffmpeg:" /tmp/versions.yaml | grep "autobuild:" | sed 's/.*"\(.*\)".*/\1/')
FFMPEG_ARCH=$(grep -A 10 "^ffmpeg:" /tmp/versions.yaml | grep "arch:" | sed 's/.*"\(.*\)".*/\1/')

if [ -z "$FFMPEG_BRANCH" ]; then
    echo "No FFmpeg branch configured, skipping."
    exit 0
fi

if [ -z "$FFMPEG_AUTOBUILD" ]; then
    echo "Error: ffmpeg.autobuild tag must be set for reproducible builds"
    exit 1
fi

# Map arch to BtbN naming convention
case "$FFMPEG_ARCH" in
    amd64) BTBN_ARCH="linux64" ;;
    arm64) BTBN_ARCH="linuxarm64" ;;
    *) echo "Error: Unsupported architecture: $FFMPEG_ARCH"; exit 1 ;;
esac

# Asset name pattern: ffmpeg-n7.1.3-43-g5a1f107b4c-linux64-gpl-7.1.tar.xz
# We need to query the release to find the exact filename for our branch and arch
RELEASE_URL="https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/${FFMPEG_AUTOBUILD}"
echo "Querying release ${FFMPEG_AUTOBUILD} for FFmpeg ${FFMPEG_BRANCH} (${FFMPEG_ARCH})..."

ASSET_INFO=$(wget -q -O - "$RELEASE_URL" | \
    grep -o '"name" *: *"[^"]*"' | \
    grep "n${FFMPEG_BRANCH}" | \
    grep "${BTBN_ARCH}-gpl-${FFMPEG_BRANCH}" | \
    grep -v "shared" | \
    head -1 | \
    sed 's/"name" *: *"//;s/"//')

if [ -z "$ASSET_INFO" ]; then
    echo "Error: Could not find FFmpeg ${FFMPEG_BRANCH} asset for ${BTBN_ARCH} in release ${FFMPEG_AUTOBUILD}"
    exit 1
fi

URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_AUTOBUILD}/${ASSET_INFO}"
echo "Downloading ${ASSET_INFO}..."
mkdir -p /utilities/ffmpeg

wget -q --show-progress \
    -O /tmp/ffmpeg.tar.xz \
    "$URL" \
    || { echo "Error: Failed to download FFmpeg from $URL"; exit 1; }

tar -xf /tmp/ffmpeg.tar.xz -C /tmp/
cp /tmp/ffmpeg-n*/bin/ffmpeg /utilities/ffmpeg/ffmpeg
cp /tmp/ffmpeg-n*/bin/ffprobe /utilities/ffmpeg/ffprobe
chmod +x /utilities/ffmpeg/ffmpeg /utilities/ffmpeg/ffprobe
rm -rf /tmp/ffmpeg*

echo "FFmpeg download complete!"
ls -lh /utilities/ffmpeg/
