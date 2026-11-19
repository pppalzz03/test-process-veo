#!/usr/bin/env bash
# Print codec, duration, fps, resolution and start offset of a video (read-only).
set -euo pipefail
f="${1:?usage: probe_video.sh file.mp4}"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,profile,width,height,avg_frame_rate,r_frame_rate,duration,start_time,nb_frames,pix_fmt \
  -show_entries format=duration,size,bit_rate -of default=noprint_wrappers=1 "$f"
