# server.py
import base64
import asyncio
import io
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import anyio
import cv2
import numpy as np
import torch
from fastapi import (
    Body, FastAPI, File, HTTPException, Query, Response, UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from sam2.build_sam import build_sam2_video_predictor


# ==============================
# App & CORS
# ==============================
app = FastAPI(title="Video → Frames Browser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 开发阶段放开；发布请收紧
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Paths
# ==============================
ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
OUTPUTS_DIR = ROOT / "outputs"
WEB_DIR = ROOT / "web"

VIDEOS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ==============================
# SAM2
# ==============================
SAM2_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAM2_CFG = str("configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_CKPT = str((ROOT / "checkpoints" / "sam2.1_hiera_large.pt").resolve())

SAM2_PREDICTOR = None              # 模型本体
SAM2_LOCK = asyncio.Lock()         # GPU 互斥
SAM2_STATES: Dict[str, Any] = {}   # { video_stem: inference_state }

def sam2_load_model():
    global SAM2_PREDICTOR
    if SAM2_PREDICTOR is None:
        print(f"[SAM2] loading on {SAM2_DEVICE}...")
        SAM2_PREDICTOR = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=SAM2_DEVICE,
                                                    hydra_overrides_extra=['++model.add_all_frames_to_correct_as_cond=true'])
        print("[SAM2] predictor ready.")

@app.on_event("startup")
def _startup_load_sam2():
    sam2_load_model()

# ==============================
# Constants
# ==============================
ALLOWED_REGIONS = {"overall", "shaft", "wrist", "gripper"}
DEFAULT_INSTANCE = 1

# ==============================
# Models
# ==============================
class ProcessRequest(BaseModel):
    video_id: str
    force: Optional[bool] = False

class SamPoint(BaseModel):
    x: int
    y: int
    label: int  # 1=positive, 0=negative

class Sam2AddPointsRequest(BaseModel):
    video_id: str
    frame_index: int
    region: Optional[str] = None
    points: List[SamPoint]

class Sam2TrackRequest(BaseModel):
    video_id: str
    region: Optional[str] = "overall"

class PromptItem(BaseModel):
    frame_index: int
    x: int
    y: int
    label: int

class SavePayload(BaseModel):
    video_id: str
    region: str
    prompts: List[PromptItem]

class KeypointItem(BaseModel):
    frame_index: int
    x: int
    y: int
    name: Optional[str] = None    # 可选名称/标签

class SaveKeypointsPayload(BaseModel):
    video_id: str
    keypoints: List[KeypointItem]

# ==============================
# Helpers: path & index
# ==============================
def _normalize_video_id(video_id: str) -> str:
    return Path(video_id).name.strip()

def _video_stem(video_id: str) -> str:
    return Path(video_id).name.rsplit(".", 1)[0]

def _video_file_from_id(video_id: str) -> Path:
    raw = _normalize_video_id(video_id)
    cand = VIDEOS_DIR / raw
    if cand.exists():
        return cand
    stem = Path(raw).stem
    if Path(raw).suffix == "":
        for ext in (".mp4", ".MP4"):
            p = VIDEOS_DIR / f"{stem}{ext}"
            if p.exists():
                return p
    low = raw.lower()
    for p in VIDEOS_DIR.iterdir():
        if p.is_file() and p.name.lower() == low:
            return p
    return cand  # 可能不存在

def _frames_dir_from_video_id(video_id: str) -> Path:
    stem = Path(_normalize_video_id(video_id)).stem
    return OUTPUTS_DIR / stem / "frames"

def _instance_dir(video_id: str, instance_id: int) -> Path:
    stem = Path(_normalize_video_id(video_id)).stem
    d = OUTPUTS_DIR / stem / f"instance{instance_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _norm_instance(v: Optional[int]) -> int:
    try:
        iv = int(v) if v is not None else DEFAULT_INSTANCE
    except Exception:
        iv = DEFAULT_INSTANCE
    return max(1, iv)

def _list_frame_files(frames_dir: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[Path] = []
    for ext in exts:
        files.extend(frames_dir.glob(ext))
    def key(p: Path):
        import re
        s = p.stem
        nums = re.findall(r"\d+", s)
        return (s, int(nums[-1])) if nums else (s, 0)
    files.sort(key=key)
    return files

def _frame_path_by_index(video_id: str, index: int) -> Path:
    frames_dir = _frames_dir_from_video_id(video_id)
    files = _list_frame_files(frames_dir)
    if not files:
        raise HTTPException(status_code=404, detail="No frames found; run /process_video first")
    if index < 0 or index >= len(files):
        raise HTTPException(status_code=416, detail=f"index out of range (0..{len(files)-1})")
    return files[index]

def _region_or_400(region: str) -> str:
    r = (region or "").strip().lower()
    if r not in ALLOWED_REGIONS:
        raise HTTPException(status_code=400, detail=f"invalid region: {region}")
    return r

# ==============================
# Frames extraction
# ==============================
def process_video_to_frames(
    video_path: Path,
    out_frames_dir: Path,
    downsample_factor: int = 2,
    new_size: Optional[Tuple[int, int]] = (640, 352),
) -> None:
    os.makedirs(out_frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Downsample factor: {downsample_factor} (effective FPS: {fps/downsample_factor:.2f})")
    print(f"  Output directory: {out_frames_dir}\n")

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % downsample_factor != 0:
            frame_count += 1
            continue

        if new_size is not None:
            frame = cv2.resize(frame, new_size)

        out_path = out_frames_dir / f"{saved_frame_count + 1:05d}.png"
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            print(f"[Error] Failed to save frame {saved_frame_count + 1}")
            frame_count += 1
            continue

        saved_frame_count += 1
        frame_count += 1

        if saved_frame_count % 50 == 0:
            progress = (frame_count / max(1, total_frames)) * 100
            print(f"Progress: {frame_count}/{total_frames} processed, {saved_frame_count} saved ({progress:.1f}%)")

    cap.release()
    print(f"\nExtraction complete! {saved_frame_count} frames saved to {out_frames_dir}")
    if fps:
        print(f"Downsampled from {fps:.2f} FPS to {fps/downsample_factor:.2f} FPS")

# ==============================
# Mask cache (in-memory) — with instance id
# ==============================
# key: (video_stem, region, instance_id)
MASK_CACHE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

def _cache_key(video_id: str, region: str, instance_id: int):
    stem = Path(video_id).name.rsplit(".", 1)[0]
    return (stem, region, int(instance_id))

def cache_put_mask(video_id: str, region: str, frame_index: int, mask_bool, instance_id: int):
    key = _cache_key(video_id, region, instance_id)
    entry = MASK_CACHE.setdefault(key, {"masks": {}, "total": 0})
    entry["masks"][int(frame_index)] = mask_bool

def cache_get_mask(video_id: str, region: str, frame_index: int, instance_id: int):
    key = _cache_key(video_id, region, instance_id)
    entry = MASK_CACHE.get(key)
    if not entry:
        return None
    return entry["masks"].get(int(frame_index))

def cache_set_total(video_id: str, region: str, total: int, instance_id: int):
    key = _cache_key(video_id, region, instance_id)
    entry = MASK_CACHE.setdefault(key, {"masks": {}, "total": 0})
    entry["total"] = int(total)

# ==============================
# Mask I/O (disk) — under instance dir
# ==============================
def _masks_dir(video_id: str, region: str, instance_id: int) -> Path:
    frames_dir = _frames_dir_from_video_id(video_id)
    out_dir = frames_dir.parent                    # outputs/<stem>
    inst_dir = out_dir / f"instance{instance_id}"  # outputs/<stem>/instanceX
    d = inst_dir / f"masks_{region}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _mask_path(video_id: str, region: str, frame_index: int, instance_id: int) -> Path:
    return _masks_dir(video_id, region, instance_id) / f"{frame_index:05d}.png"

def _save_mask_png(video_id: str, region: str, frame_index: int, mask: np.ndarray, instance_id: int):
    """保存二值 mask（磁盘 0/255 灰度；内存缓存 bool）"""
    m = (mask > 0).astype(np.uint8) * 255
    p = _mask_path(video_id, region, frame_index, instance_id)
    Image.fromarray(m).save(p)
    cache_put_mask(video_id, region, frame_index, (m > 127), instance_id)

def _load_mask_png(video_id: str, region: str, frame_index: int, instance_id: int) -> Optional[np.ndarray]:
    m = cache_get_mask(video_id, region, frame_index, instance_id)
    if m is not None:
        return m
    p = _mask_path(video_id, region, frame_index, instance_id)
    if not p.exists():
        return None
    m = (np.array(Image.open(p).convert("L")) > 127)
    cache_put_mask(video_id, region, frame_index, m, instance_id)
    return m

# ==============================
# Image helpers
# ==============================
def _read_frame_as_rgb(video_id: str, index: int) -> np.ndarray:
    path = _frame_path_by_index(video_id, index)
    img = Image.open(path).convert("RGB")
    return np.array(img)

def _overlay_mask_on_image(img_rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    overlay = img_rgb.copy()
    c = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + c * alpha).astype(np.uint8)
    return overlay

def _to_b64_png(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _mask_bool_to_rgba_bytes(mask_bool: np.ndarray,
                             color=(0, 255, 0),  # RGB
                             alpha=128) -> bytes:
    """把二值 mask 转透明叠加 RGBA PNG（前景=颜色+alpha，背景=全透明）"""
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    fg = mask_bool
    rgba[..., 0][fg] = color[0]
    rgba[..., 1][fg] = color[1]
    rgba[..., 2][fg] = color[2]
    rgba[..., 3][fg] = np.uint8(max(0, min(255, int(alpha))))
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()

# ==============================
# SAM2 state per video
# ==============================
async def sam2_init_for_video(video_id: str):
    stem = _video_stem(video_id)
    frames_dir = _frames_dir_from_video_id(video_id)
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="frames_dir not found; run /process_video first")
    async with SAM2_LOCK:
        state = SAM2_PREDICTOR.init_state(video_path=str(frames_dir))
        SAM2_PREDICTOR.reset_state(state)
        SAM2_STATES[stem] = state
        print(f"[SAM2] init_state ok: video={stem}, frames_dir={frames_dir}")

async def sam2_reset_for_video(video_id: str):
    if SAM2_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="SAM2 predictor not loaded")
    stem = _video_stem(video_id)
    state = SAM2_STATES.get(stem)
    if state is None:
        await sam2_init_for_video(video_id)
        state = SAM2_STATES.get(stem)
        if state is None:
            raise HTTPException(status_code=500, detail="SAM2 state missing after init")
    async with SAM2_LOCK:
        SAM2_PREDICTOR.reset_state(state)
        print(f"[SAM2] reset_state done for video={stem}")

# ==============================
# Tracking jobs (include instance)
# ==============================
# key: (stem, region, instance_id)
TRACK_JOBS: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

# ==============================
# Routes
# ==============================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/videos")
def list_videos():
    if not VIDEOS_DIR.exists():
        return {"videos": []}
    vids = set()
    vids.update(p.name for p in VIDEOS_DIR.glob("*.mp4"))
    vids.update(p.name for p in VIDEOS_DIR.glob("*.MP4"))
    return {"videos": sorted(vids)}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    name_lower = file.filename.lower()
    if not (name_lower.endswith(".mp4") or name_lower.endswith(".MP4")):
        raise HTTPException(status_code=400, detail="Only .mp4 is accepted")
    dst = VIDEOS_DIR / file.filename
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"video_id": file.filename, "saved_to": str(dst)}

@app.post("/process_video")
def process_video(req: ProcessRequest):
    video_path = _video_file_from_id(req.video_id)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {Path(req.video_id).name}")

    frames_dir = _frames_dir_from_video_id(req.video_id)
    need_run = req.force or (not frames_dir.exists())
    if need_run:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        process_video_to_frames(video_path, frames_dir)

    files = _list_frame_files(frames_dir)
    if not files:
        raise HTTPException(status_code=500, detail="No frames generated")

    w = h = None
    try:
        with Image.open(files[0]) as im:
            w, h = im.size
    except Exception:
        pass

    try:
        anyio.from_thread.run(sam2_init_for_video, req.video_id)
    except Exception as e:
        print(f"[SAM2] init_state failed (lazy init later): {e}")

    return {
        "video_id": Path(req.video_id).name,
        "frames_dir": str(frames_dir.relative_to(ROOT)),
        "num_frames": len(files),
        "frame_size": [w, h] if (w and h) else None,
    }

@app.get("/frames_meta")
def frames_meta(video_id: str = Query(...)):
    frames_dir = _frames_dir_from_video_id(video_id)
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames not found, run /process_video first")
    files = _list_frame_files(frames_dir)
    if not files:
        raise HTTPException(status_code=404, detail="No frames found in frames_dir")
    w = h = None
    try:
        with Image.open(files[0]) as im:
            w, h = im.size
    except Exception:
        pass
    return {
        "video_id": Path(video_id).name,
        "num_frames": len(files),
        "frame_size": [w, h] if (w and h) else None,
    }

@app.post("/sam2/reset")
async def sam2_reset(video_id: str = Body(..., embed=True)):
    """前端切换 region / instance 时调用即可。"""
    await sam2_reset_for_video(video_id)
    return {"ok": True}

@app.get("/frame")
def get_frame(video_id: str = Query(...), index: int = Query(..., ge=0)):
    path = _frame_path_by_index(video_id, index)
    ext = path.suffix.lower()
    mime = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp":  "image/bmp",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    resp = FileResponse(path, media_type=mime)
    # 强缓存：浏览器可用 version query 破缓存
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    st = path.stat()
    resp.headers["ETag"] = f'W/"{st.st_mtime_ns}-{st.st_size}"'
    return resp

# ============= Points =============
@app.post("/save_points")
def save_points(payload: SavePayload,
                append: bool = Query(False),
                region: Optional[str] = Query(None),
                instance_id: int = Query(DEFAULT_INSTANCE)):
    """保存到 outputs/<stem>/instanceX/points_<region>.json"""
    instance_id = _norm_instance(instance_id)
    body_region = _region_or_400(payload.region)
    if region is not None:
        query_region = _region_or_400(region)
        if query_region != body_region:
            raise HTTPException(status_code=400, detail="region mismatch between query and body")

    vid = Path(payload.video_id).name
    out_root = _instance_dir(vid, instance_id)
    out_path = out_root / f"points_{body_region}.json"

    data_to_save = {
        "video_id": vid,
        "instance_id": instance_id,
        "region": body_region,
        "num_prompts": len(payload.prompts),
        "prompts": [p.dict() for p in payload.prompts],
    }

    if append and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as r:
                old = json.load(r)
        except Exception:
            old = {}
        old_prompts = old.get("prompts", [])
        old_prompts.extend(data_to_save["prompts"])
        data_to_save["prompts"] = old_prompts
        data_to_save["num_prompts"] = len(old_prompts)

    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(data_to_save, w, ensure_ascii=False, indent=2)

    return {"saved": data_to_save["num_prompts"], "path": str(out_path)}

@app.get("/load_points")
def load_points(video_id: str = Query(...),
                region: str = Query("overall"),
                instance_id: int = Query(DEFAULT_INSTANCE)):
    """读取 outputs/<stem>/instanceX/points_<region>.json"""
    instance_id = _norm_instance(instance_id)
    vid = Path(video_id).name
    r = _region_or_400(region)
    out_root = _instance_dir(vid, instance_id)
    points_path = out_root / f"points_{r}.json"
    if not points_path.exists():
        raise HTTPException(status_code=404, detail=f"{points_path.name} not found for video_id={vid}, instance={instance_id}")
    try:
        with open(points_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read {points_path.name}: {e}")
    return data

# ============= Keypoints =============
@app.post("/save_keypoints")
def save_keypoints(payload: SaveKeypointsPayload,
                   instance_id: int = Query(DEFAULT_INSTANCE),
                   append: bool = Query(True)):
    """
    保存关键点到 outputs/<stem>/instanceX/keypoints.json
    自动补充 frame_filename。
    """
    instance_id = _norm_instance(instance_id)
    vid = Path(payload.video_id).name
    out_root = _instance_dir(vid, instance_id)
    out_path = out_root / "keypoints.json"

    # 组装并补全 frame_filename
    enriched = []
    for kp in payload.keypoints:
        try:
            fpath = _frame_path_by_index(vid, kp.frame_index)
            fname = fpath.name
        except HTTPException:
            fname = None
        row = {
            "frame_index": kp.frame_index,
            "frame_filename": fname,
            "x": kp.x,
            "y": kp.y,
            "name": kp.name,
        }
        enriched.append(row)

    data_to_save = {
        "video_id": vid,
        "instance_id": instance_id,
        "num_keypoints": len(enriched),
        "keypoints": enriched,
    }

    if append and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as r:
                old = json.load(r)
        except Exception:
            old = {}
        old_kps = old.get("keypoints", [])
        old_kps.extend(enriched)
        data_to_save["keypoints"] = old_kps
        data_to_save["num_keypoints"] = len(old_kps)

    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(data_to_save, w, ensure_ascii=False, indent=2)

    return {"saved": data_to_save["num_keypoints"], "path": str(out_path)}

@app.get("/load_keypoints")
def load_keypoints(video_id: str = Query(...),
                   instance_id: int = Query(DEFAULT_INSTANCE)):
    """读取 outputs/<stem>/instanceX/keypoints.json"""
    instance_id = _norm_instance(instance_id)
    vid = Path(video_id).name
    out_root = _instance_dir(vid, instance_id)
    kp_path = out_root / "keypoints.json"
    if not kp_path.exists():
        raise HTTPException(status_code=404, detail=f"keypoints.json not found for video_id={vid}, instance={instance_id}")
    try:
        with open(kp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read keypoints.json: {e}")
    return data

# ============= Masks (PNG / RGBA overlay) =============
@app.get("/mask")
def get_mask(
    video_id: str = Query(...),
    frame_index: int = Query(..., ge=0),
    region: str = Query("overall"),
    instance_id: int = Query(DEFAULT_INSTANCE),
    rgba: int = Query(0),
    a: int = Query(128),
    r: int = Query(0),
    g: int = Query(255),
    b: int = Query(0),
):
    instance_id = _norm_instance(instance_id)
    region = _region_or_400(region)
    m = _load_mask_png(video_id, region, frame_index, instance_id)
    if m is None:
        raise HTTPException(status_code=404, detail="mask not found")

    if rgba:
        try:
            data = _mask_bool_to_rgba_bytes(m.astype(bool), color=(r, g, b), alpha=a)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"rgba encode failed: {e}")
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        return Response(content=data, media_type="image/png", headers=headers)

    p = _mask_path(video_id, region, frame_index, instance_id)
    resp = FileResponse(p, media_type="image/png")
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp

@app.head("/mask")
def head_mask(video_id: str = Query(...),
              frame_index: int = Query(..., ge=0),
              region: str = Query("overall"),
              instance_id: int = Query(DEFAULT_INSTANCE)):
    instance_id = _norm_instance(instance_id)
    region = _region_or_400(region)
    p = _mask_path(video_id, region, frame_index, instance_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="mask not found")
    resp = Response(status_code=200)
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    resp.headers["Content-Type"] = "image/png"
    try:
        st = p.stat()
        resp.headers["Content-Length"] = str(st.st_size)
    except Exception:
        pass
    return resp

# ============= SAM2 add_points / track / render =============
@app.post("/sam2/add_points")
async def sam2_add_points(req: Sam2AddPointsRequest,
                          instance_id: int = Query(DEFAULT_INSTANCE)):
    """点击交互后，更新该帧 mask（保存到 instanceX），并返回 overlay/mask 的 base64 PNG"""
    instance_id = _norm_instance(instance_id)

    if SAM2_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="SAM2 predictor not loaded")

    stem = _video_stem(req.video_id)
    if stem not in SAM2_STATES:
        await sam2_init_for_video(req.video_id)
    state = SAM2_STATES.get(stem)
    if state is None:
        raise HTTPException(status_code=500, detail="SAM2 state missing after init")

    img_rgb = _read_frame_as_rgb(req.video_id, req.frame_index)

    pts = np.array([[p.x, p.y] for p in req.points], dtype=np.float32) if req.points else np.zeros((0, 2), np.float32)
    lbls = np.array([p.label for p in req.points], dtype=np.int32) if req.points else np.zeros((0,), np.int32)

    # 简单二分类对象 id 固定为 1；如需 region->obj_id，可替换为你的映射
    ann_obj_id = 1

    async with SAM2_LOCK:
        try:
            _, out_obj_ids, out_mask_logits = SAM2_PREDICTOR.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(req.frame_index),
                obj_id=int(ann_obj_id),
                points=pts if len(pts) else None,
                labels=lbls if len(lbls) else None,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SAM2 add_points failed: {e}")

    logits = out_mask_logits[0][0].detach().cpu().numpy()
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    mask_logit = np.asarray(logits)
    if mask_logit.ndim == 3:
        mask_logit = mask_logit[0]
    mask = (mask_logit > 0).astype(np.uint8)

    _save_mask_png(req.video_id, (req.region or "overall"), req.frame_index, mask, instance_id)

    overlay_rgb = _overlay_mask_on_image(img_rgb, mask, color=(0, 255, 0), alpha=0.4)
    mask_rgb = np.stack([(mask * 255).astype(np.uint8)] * 3, axis=-1)

    return {
        "video_id": req.video_id,
        "frame_index": req.frame_index,
        "region": (req.region or "overall"),
        "instance_id": instance_id,
        "obj_id": ann_obj_id,
        "mask_area": int(mask.sum()),
        "overlay_png_b64": _to_b64_png(overlay_rgb),
        "mask_png_b64": _to_b64_png(mask_rgb),
    }

@app.post("/sam2/track/start")
async def sam2_track_start(req: Sam2TrackRequest,
                           instance_id: int = Query(DEFAULT_INSTANCE)):
    """启动整段视频的 mask 传播，结果写入 instanceX 的 masks_*. """
    instance_id = _norm_instance(instance_id)

    if SAM2_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="SAM2 predictor not loaded")

    region = _region_or_400(req.region or "overall")
    stem = _video_stem(req.video_id)

    frames_dir = _frames_dir_from_video_id(req.video_id)
    files = _list_frame_files(frames_dir)
    if not files:
        raise HTTPException(status_code=404, detail="No frames found; run /process_video first")
    total = len(files)

    key = (stem, region, instance_id)
    TRACK_JOBS[key] = {"state": "running", "current": 0, "total": total, "message": ""}

    cache_set_total(req.video_id, region, total, instance_id)

    async def _runner():
        try:
            if stem not in SAM2_STATES:
                await sam2_init_for_video(req.video_id)
            state = SAM2_STATES.get(stem)
            if state is None:
                raise RuntimeError("SAM2 state missing after init")

            async with SAM2_LOCK:
                for out_frame_idx, out_obj_ids, out_mask_logits in SAM2_PREDICTOR.propagate_in_video(state):
                    logits = out_mask_logits
                    if hasattr(logits, "detach"):
                        logits = logits.detach().cpu()
                    if hasattr(logits, "numpy"):
                        logits = logits.numpy()

                    if logits.ndim == 4:      # [N, 1, H, W]
                        mask_logit = logits[0, 0]
                    elif logits.ndim == 3:    # [1, H, W] or similar
                        mask_logit = logits[0] if logits.shape[0] in (1,) else logits.squeeze()
                    else:                      # [H, W]
                        mask_logit = logits

                    mask = (mask_logit > 0).astype(np.uint8)
                    _save_mask_png(req.video_id, region, int(out_frame_idx), mask, instance_id)

                    TRACK_JOBS[key]["current"] = int(out_frame_idx) + 1

            TRACK_JOBS[key]["state"] = "done"
            TRACK_JOBS[key]["message"] = "ok"

        except Exception as e:
            TRACK_JOBS[key]["state"] = "error"
            TRACK_JOBS[key]["message"] = str(e)

    asyncio.create_task(_runner())
    return {"ok": True, "total": total}

@app.get("/sam2/track/status")
def sam2_track_status(video_id: str = Query(...),
                      region: str = Query("overall"),
                      instance_id: int = Query(DEFAULT_INSTANCE)):
    instance_id = _norm_instance(instance_id)
    stem = _video_stem(video_id)
    key = (stem, _region_or_400(region), instance_id)
    job = TRACK_JOBS.get(key)
    if not job:
        return {"state": "idle", "current": 0, "total": 0, "message": ""}
    return job

@app.get("/sam2/render")
def sam2_render(video_id: str = Query(...),
                frame_index: int = Query(..., ge=0),
                region: str = Query("overall"),
                instance_id: int = Query(DEFAULT_INSTANCE)):
    instance_id = _norm_instance(instance_id)
    region = _region_or_400(region)
    m = _load_mask_png(video_id, region, frame_index, instance_id)
    if m is None:
        return Response(status_code=204)
    img_rgb = _read_frame_as_rgb(video_id, frame_index)
    overlay_rgb = _overlay_mask_on_image(img_rgb, m.astype(np.uint8), color=(0, 255, 0), alpha=0.5)
    mask_rgb = np.stack([(m.astype(np.uint8) * 255)] * 3, axis=-1)
    return {
        "video_id": video_id,
        "frame_index": frame_index,
        "region": region,
        "instance_id": instance_id,
        "overlay_png_b64": _to_b64_png(overlay_rgb),
        "mask_png_b64": _to_b64_png(mask_rgb),
    }

# ==============================
# Web (static)
# ==============================
@app.get("/", include_in_schema=False)
async def root():
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail=f"{index_file} not found")
    return FileResponse(index_file)

# 可直接访问整个 web 目录
app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
