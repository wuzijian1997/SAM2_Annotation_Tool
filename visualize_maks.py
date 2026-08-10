#!/usr/bin/env python3
"""
visualize_masks.py

Overlay segmentation masks from an instance folder onto a video frame.

Usage:
    python visualize_masks.py --video_root /path/to/surgpose_data --video video_1 --frame 1 --instance instance1 --out out.png --show

The script expects masks directories inside the instance folder with names like
  masks_shaft/, masks_wrist/, masks_gripper/, masks_overall/
Each masks directory should contain PNG masks matching the frame filename (e.g. 00001.png).

If JSON point files exist (points_*.json), they're ignored by default.

"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import sys

PART_COLORS = {
    'gripper': (0, 0, 255),   # red
    'shaft':   (0, 255, 0),   # green
    'wrist':   (255, 0, 0),   # blue
    'overall': (0, 255, 255), # yellow
}


def parse_args():
    p = argparse.ArgumentParser(description='Visualize segmentation masks overlayed on a frame')
    p.add_argument('--video_root', default='/mnt/data0/shuojue/jiaao-temp/sam2_annotation/outputs', help='Root folder containing video folders')
    p.add_argument('--video', default = 'SARRARP502022_needleGrasping_161_video15', help='Video folder name (e.g. video_1)')
    p.add_argument('--frame', default='00001.png', help='Frame index or filename (e.g. 1 or 00001.png)')
    p.add_argument('--instance', default='instance2', help='Instance folder name inside the video folder')
    p.add_argument('--alpha', type=float, default=0.5, help='Alpha blending for masks (0-1)')
    p.add_argument('--out', default = './vis_figure', help='Output path to save visualization (PNG). If omitted, will show window only')
    p.add_argument('--show', action='store_true', help='Show visualization in an OpenCV window')
    p.add_argument('--video-out', default = './', help='Output MP4 path to write visualization across frames')
    p.add_argument('--fps', type=int, default=20, help='FPS for output video')
    p.add_argument('--start', type=int, default=1, help='Start frame index (1-based)')
    p.add_argument('--end', type=int, default=0, help='End frame index (inclusive). 0 means last available frame')
    p.add_argument('--mask-offset', type=int, default=-1, help='If masks numbering is shifted relative to frames (masks start at 0 while frames start at 1), set offset (mask_index = frame_index + offset)')
    p.add_argument('--refined', action='store_true', help='Use refined_masks_*/ directories instead of masks_*/')
    return p.parse_args()


def frame_filename(frame_arg):
    # Accept either numeric index or filename
    s = str(frame_arg)
    if s.isdigit():
        return f"{int(s):05d}.png"
    return s


def find_mask_dirs(instance_dir: Path, refined: bool=False):
    # Return mapping part_name -> mask_dir Path
    mask_dirs = {}
    if not refined:
        for p in instance_dir.iterdir():
            if p.is_dir() and p.name.startswith('masks_'):
                part = p.name[len('masks_'):]
                mask_dirs[part] = p
        return mask_dirs
    else:
        # look for refined masks
        for p in instance_dir.iterdir():
            if p.is_dir() and p.name.startswith('refined_masks_'):
                part = p.name[len('refined_masks_'):]
                mask_dirs[part] = p
        return mask_dirs


def load_mask(mask_path: Path, shape):
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    # If 3 channels, convert to gray by taking any channel
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    # Resize if needed
    if (m.shape[0], m.shape[1]) != (shape[0], shape[1]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    # Normalize to binary mask
    _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m_bin


def overlay_masks(image: np.ndarray, masks: dict, alpha: float=0.5):
    # image: HxWx3 BGR uint8
    overlay = image.copy().astype(np.float32)
    out = image.copy().astype(np.float32)
    h, w = image.shape[:2]

    for part, mask in masks.items():
        if mask is None:
            continue
        if part == 'overall':
            continue
        color = PART_COLORS.get(part, None)
        if color is None:
            # pick a pseudo-random color based on hash
            rng = abs(hash(part))
            color = ((rng >> 0) & 255, (rng >> 8) & 255, (rng >> 16) & 255)
        # create colored layer
        colored = np.zeros_like(image, dtype=np.float32)
        colored[:, :] = color
        # mask normalized 0..1
        m = (mask.astype(np.float32) / 255.0)[:, :, None]
        # blend
        out = out * (1 - m * alpha) + colored * (m * alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def main():
    args = parse_args()
    video_root = Path(args.video_root)
    video_dir = video_root / args.video
    if not video_dir.exists():
        print(f"Video folder not found: {video_dir}")
        sys.exit(1)
    instance_dir = video_dir / args.instance
    if not instance_dir.exists():
        print(f"Instance folder not found: {instance_dir}")
        sys.exit(1)
    # list frame files
    frames_dir = video_dir / 'frames'
    if not frames_dir.exists():
        print(f"Frames folder not found: {frames_dir}")
        sys.exit(1)

    all_frames = sorted([p.name for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.png', '.jpg']])
    if not all_frames:
        print(f"No frames found in: {frames_dir}")
        sys.exit(1)

    # determine start/end indices
    start_idx = max(1, args.start)
    if args.end > 0:
        end_idx = args.end
    else:
        # infer from available frames
        # frames are expected to be numbered 00001.png ... so take max
        try:
            end_idx = max(int(p.stem) for p in frames_dir.iterdir() if p.is_file())
        except Exception:
            end_idx = len(all_frames)

    # find mask dirs
    mask_dirs = find_mask_dirs(instance_dir, refined=args.refined)
    if not mask_dirs:
        print(f"No mask directories found in: {instance_dir}")

    # If video-out requested, prepare writer after reading first frame to get size
    writer = None
    if args.video_out:
        # read first frame to get size
        first_frame_name = f"{start_idx:05d}.png"
        first_frame_path = frames_dir / first_frame_name
        if not first_frame_path.exists():
            # fallback to first available
            first_frame_path = frames_dir / all_frames[0]
        img0 = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if img0 is None:
            print(f"Failed to read initial frame for video writer: {first_frame_path}")
            sys.exit(1)
        h, w = img0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outp = Path(args.video_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(outp), fourcc, float(args.fps), (w, h))
        if not writer.isOpened():
            print(f"Failed to open video writer for {outp}")
            writer = None

    # iterate frames
    for fi in range(start_idx, end_idx + 1):
        frame_name = f"{fi:05d}.png"
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            # warn and skip
            print(f"Warning: frame missing {frame_path}, skipping")
            continue
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to read frame {frame_path}, skipping")
            continue

        masks = {}
        # masks may be offset: mask_index = frame_index + mask_offset
        mask_index = fi + args.mask_offset
        mask_filename = f"{mask_index:05d}.png"
        for part, d in mask_dirs.items():
            mask_path = d / mask_filename
            masks[part] = load_mask(mask_path, img.shape[:2])

        vis = overlay_masks(img, masks, alpha=args.alpha)

        if args.video_out and writer is not None:
            writer.write(vis)

        # if showing, display each frame
        # if args.show:
        #     cv2.imshow('visualization', vis)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # optionally save individual frame
        if args.out:
            out_frame_path = Path(args.out)
            # if out is a directory, write each frame inside
            if out_frame_path.suffix == '':
                out_frame_path = out_frame_path / frame_name
            else:
                # treat as filename pattern? write once for first frame
                if fi != start_idx:
                    pass
            out_frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_frame_path), vis)

    if writer is not None:
        writer.release()
        print(f"Saved video to {args.video_out}")

    # if args.show:
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
