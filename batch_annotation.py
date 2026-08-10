from sam2.build_sam import build_sam2_video_predictor
import torch
from pathlib import Path
import os
import json
from PIL import Image
import numpy as np
import cv2



def read_points_json(points_json_path):
    """Read a points JSON (points_{part}.json) and return a mapping from
    frame_index -> list of point prompts, and a sorted list of annotated frame indices.

    Expected JSON shape (example):
    {
      "video_id": "...",
      "instance_id": 1,
      "region": "gripper",
      "num_prompts": 5,
      "prompts": [ {"frame_index":0, "x":321, "y":160, "label":1}, ...]
    }

    Returns:
      frames_map: dict[int, list[dict]]  -- e.g. {0: [{"x":321,"y":160,"label":1}, ...], 9: [...]}
      annotated_frames: list[int] -- sorted unique frame indices that have prompts
    """
    try:
        with open(points_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read points json '{points_json_path}': {e}")
        return {}, []

    prompts = data.get('prompts', [])
    frames_map = {}
    for p in prompts:
        # be tolerant to strings vs ints
        try:
            fi = int(p.get('frame_index'))
        except Exception:
            continue
        x = p.get('x')
        y = p.get('y')
        label = p.get('label')
        frames_map.setdefault(fi, []).append({'x': x, 'y': y, 'label': label})

    annotated_frames = sorted(frames_map.keys())
    return frames_map, annotated_frames
def _mask_path_for_frame(masks_dir, frame_index):
    return os.path.join(masks_dir, f"{int(frame_index):05d}.png")

def _load_mask_png_as_bool(pth):
    if not os.path.exists(pth):
        return None
    try:
        im = Image.open(pth).convert("L")
        arr = np.array(im)
        return (arr > 127)
    except Exception as e:
        print(f"Failed to load mask '{pth}': {e}")
        return None

def propagate_with_reference_masks(frames_dir, masks_dir, annotated_frames, out_base_dir, predictor, state, visualization = False):
    """Use annotated frame masks (in masks_dir) as conditioning masks and
    propagate across the whole video with `predictor`.

    Saves results to out_base_dir/masks_{part_name}/{frame:05d}.png
    Returns number of frames written.
    """
    # initialize predictor state for this video
    
    frame_names = sorted([p.name for p in Path(frames_dir).iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    predictor.reset_state(state)

    ann_obj_id = 1

    # add all annotated masks as mask inputs
    added_any = False
    for fi in annotated_frames:
        mpath = _mask_path_for_frame(masks_dir, fi)
        mask_bool = _load_mask_png_as_bool(mpath)
        if mask_bool is None:
            print(f"Reference mask not found for frame {fi} at {mpath}; skipping this reference frame")
            continue
     
        predictor.add_new_mask(inference_state=state, frame_idx=int(fi), obj_id=ann_obj_id, mask=mask_bool)
        added_any = True
        print(f"Added reference mask for frame {fi}")


    if not added_any:
        print("No valid reference masks found; skipping propagation")
        return 0

    # ensure output dir
    out_masks_dir = os.path.join(out_base_dir)
    os.makedirs(out_masks_dir, exist_ok=True)

    written = 0

    def _save_from_generator(gen):
        nonlocal written
        for out_frame_idx, out_obj_ids, out_mask_logits in gen:
            logits = out_mask_logits
            if hasattr(logits, 'detach'):
                logits = logits.detach().cpu()
            if hasattr(logits, 'numpy'):
                logits = logits.numpy()

            # reshape / select likely shape
            if logits.ndim == 4:  # [N, 1, H, W]
                mask_logit = logits[0, 0]
            elif logits.ndim == 3:  # [N, H, W] or [1, H, W]
                mask_logit = logits[0] if logits.shape[0] in (1,) else logits.squeeze()
            else:
                mask_logit = logits

            mask = (mask_logit > 0).astype(np.uint8) * 255
            out_p = os.path.join(out_masks_dir, f"{int(out_frame_idx):05d}.png")
            # avoid overwriting existing file (optional)
            if os.path.exists(out_p):
                # skip writing if already present
                continue
            try:
                Image.fromarray(mask).save(out_p)
                written += 1
            except Exception as e:
                print(f"Failed to save propagated mask to {out_p}: {e}")
            if visualization == True:
                visualization_dir = 'visualization'
                if not os.path.exists(visualization_dir):
                    os.makedirs(visualization_dir)
                vis_p = os.path.join(visualization_dir, frame_names[int(out_frame_idx)])
                # alpha blending to overlay mask on original image; read original image from frames_dir
                frame_p = os.path.join(frames_dir, frame_names[int(out_frame_idx)])

                orig_bgr = cv2.imread(frame_p, cv2.IMREAD_COLOR)
                if orig_bgr is None:
                    print(f"Failed to read frame {frame_p}")
                else:
                    orig = orig_bgr.astype(np.float32) / 255.0
                    mask_alpha = mask[..., np.newaxis]/255.0 * 0.5
                    # red overlay in BGR
                    red_bgr = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    vis = orig * (1.0 - mask_alpha) + red_bgr * mask_alpha
                    vis_bgr = (vis * 255.0).astype(np.uint8)
                    cv2.imwrite(vis_p, vis_bgr)

    # First: propagate in reverse from the earliest annotated frame to frame 0
    start_frame = min(annotated_frames)

    if start_frame > 0:
        rev_gen = predictor.propagate_in_video(state, start_frame_idx=start_frame, max_frame_num_to_track=start_frame + 1, reverse=True)
        _save_from_generator(rev_gen)

    fwd_gen = predictor.propagate_in_video(state, start_frame_idx=start_frame, reverse=False)
    _save_from_generator(fwd_gen)


    print(f"Propagation complete, saved {written} masks to {out_masks_dir}")
    return written
    
if __name__ == "__main__":
    videos_path = '/mnt/data0/shuojue/jiaao-temp/sam2_annotation/videos'
    annotations_dir = videos_path.replace('videos', 'outputs')
    outputs_dir = 'refined_outputs'

    ROOT = Path(__file__).resolve().parent
    SAM2_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAM2_CFG = str("configs/sam2.1/sam2.1_hiera_b+.yaml")
    SAM2_CKPT = str((ROOT / "checkpoints" / "checkpoint.pt").resolve())
    SAM2_PREDICTOR = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=SAM2_DEVICE,
                                                        hydra_overrides_extra=['++model.add_all_frames_to_correct_as_cond=true'])


    video_names = [v for v in os.listdir(videos_path) if 'SARRARP' in v]
    for video_name in video_names:
        video_full_path = os.path.join(videos_path, video_name)
        annotation_full_path = os.path.join(annotations_dir, video_name.replace('.mp4', ''))
        
        if not os.path.exists(annotation_full_path):
            print(f"Annotation for {video_name} has not been created yet. Skipping...")
            continue
        instance1_dir = os.path.join(annotation_full_path, 'instance1')
        instance2_dir = os.path.join(annotation_full_path, 'instance2')
        if os.path.exists(instance2_dir) == True:
            masks = os.listdir(instance2_dir)
            masks = [m for m in masks if 'refined_masks_' in m]
            if len(masks) == 4:
                print(f"Refined masks for all parts in instance 2 of {video_name} already exist. Skipping...")
                continue
        try:
            frames_dir = os.path.join(annotation_full_path, 'frames')
            state = SAM2_PREDICTOR.init_state(video_path=str(frames_dir))
        except Exception as e:
            print(f"Failed to initialize predictor state for {video_name}: {e}. Skipping...")
            continue
        
        for i, instance_dir in enumerate([instance1_dir, instance2_dir], start=1):
            kpt_json_path   = os.path.join(instance_dir, 'keypoints.json')
            if not os.path.exists(kpt_json_path):
                print(f"Keypoint JSON for {video_name} instance {i} not found. Skipping...")
                continue
            part_names = ['overall', 'gripper', 'shaft', 'wrist']
            for part_name in part_names:
                masks_dir = os.path.join(instance_dir, f'masks_{part_name}')
                points_json = os.path.join(instance_dir, f'points_{part_name}.json')
                if not os.path.exists(masks_dir):
                    print(f"Mask directory for {video_name} instance {i} part {part_name} not found. Skipping...")
                    continue
                if not os.path.exists(points_json):
                    print(f"Points prompts JSON for {video_name} instance {i} part {part_name} not found. Skipping...")
                    continue
                # read points_json and prepare prompts
                frames_map, annotated_frames = read_points_json(points_json)
                if not annotated_frames:
                    print(f"No annotated frames found in {points_json} for {video_name} instance {i} part {part_name}. Skipping...")
                    continue

                # annotated_frames contains the frame indices which had point prompts
                # These are the frames where the existing segmentation masks (from point prompts)
                # are expected to be most accurate and therefore should be used as reference masks.
                print(f"Video {video_name} instance {i} part {part_name} - annotated frames: {annotated_frames}")



                # run propagation using the annotated frames' masks as reference
                refined_out_root = os.path.join(instance_dir, f"refined_masks_{part_name}")
                propagate_with_reference_masks(frames_dir, masks_dir, annotated_frames, refined_out_root, SAM2_PREDICTOR,state, True)
                # frames_map is a dict mapping each annotated frame index to the list of point prompts
                # (you can use frames_map[frame_index] if you want the exact points for that frame)
                # At this point we only collect the frame indices. The next step is to locate the
                # corresponding mask files in `masks_dir` (naming depends on how masks were saved)
                # and treat those masks as reference masks for propagation with the new SAM2 model.