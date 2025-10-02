#!/usr/bin/env python3
"""High-quality parallax animation for single images.

This script focuses solely on depth-based camera motion with layered reprojection
and edge-aware processing tuned for high-end GPUs such as the RTX 4090.  It pairs
Transformers' Depth Anything v2 depth with multi-layer warping, gradient-aware
dampening, and inpainted backgrounds for clean occlusion reveals.  Typical 1080p
renders finish within ~30–40 seconds on an RTX 4090.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Prefer Tensor Cores for depth while keeping FP32 accumulation quality
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_HAS_GUIDED_FILTER = bool(getattr(cv2, "ximgproc", None)) and hasattr(cv2.ximgproc, "guidedFilter")

# --- Color space helpers (sRGB <-> linear) ---
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a).astype(np.float32)

# ----------------------------
# Utilities
# ----------------------------

@dataclass
class DeviceConfig:
    torch_device: torch.device
    use_half: bool


def pick_device(preference: str = "cuda") -> DeviceConfig:
    if preference == "cuda" and torch.cuda.is_available():
        return DeviceConfig(torch.device("cuda"), use_half=True)
    return DeviceConfig(torch.device("cpu"), use_half=False)


def load_image(path: Path, max_dim: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb)  # work in linear-light


# ----------------------------
# Depth estimation
# ----------------------------

_model_cache: dict[tuple[str, torch.dtype], AutoModelForDepthEstimation] = {}
_processor_cache: dict[str, AutoImageProcessor] = {}


def get_depth_model(name: str, device: DeviceConfig) -> tuple[AutoModelForDepthEstimation, AutoImageProcessor]:
    dtype = torch.float16 if device.use_half else torch.float32
    cache_key = (name, dtype)
    processor = _processor_cache.get(name)
    if processor is None:
        processor_kwargs = {}
        try:
            if importlib.util.find_spec("torchvision") is not None:
                processor_kwargs["use_fast"] = True
        except ModuleNotFoundError:
            pass
        try:
            processor = AutoImageProcessor.from_pretrained(name, **processor_kwargs)
        except TypeError:
            processor = AutoImageProcessor.from_pretrained(name)
        _processor_cache[name] = processor

    if cache_key not in _model_cache:
        model = AutoModelForDepthEstimation.from_pretrained(name, dtype=dtype)
        model.to(device.torch_device)
        model.eval()
        _model_cache[cache_key] = model
    else:
        model = _model_cache[cache_key]
        model.to(device.torch_device, dtype=dtype)

    return model, processor


def estimate_depth(
    rgb: np.ndarray,
    device: DeviceConfig,
    model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
    depth_max_dim: int = 1024,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, depth_max_dim / float(max(h, w)))
    if scale < 1.0:
        small_lin = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), cv2.INTER_LANCZOS4)
    else:
        small_lin = rgb

    small = np.clip(linear_to_srgb(np.clip(small_lin, 0.0, 1.0)), 0.0, 1.0)

    model, processor = get_depth_model(model_name, device)
    inputs = processor(images=(small * 255.0).astype(np.uint8), return_tensors="pt")
    inputs = {k: v.to(device.torch_device) for k, v in inputs.items()}
    inputs_flip = processor(images=(small[:, ::-1] * 255.0).astype(np.uint8), return_tensors="pt")
    inputs_flip = {k: v.to(device.torch_device) for k, v in inputs_flip.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        outputs_flip = model(**inputs_flip)

    def _depth_from(output) -> np.ndarray:
        if hasattr(processor, "post_process_depth_estimation"):
            post_processed = processor.post_process_depth_estimation(
                output,
                target_sizes=[(small.shape[0], small.shape[1])],
            )
            return post_processed[0]["predicted_depth"].squeeze().float().cpu().numpy()
        return output.predicted_depth.squeeze().float().cpu().numpy()

    depth_small = 0.5 * (
        _depth_from(outputs) + _depth_from(outputs_flip)[:, ::-1]
    )

    depth = cv2.resize(depth_small, (w, h), cv2.INTER_CUBIC)
    depth = np.clip(depth, 0, None).astype(np.float32)
    lo, hi = np.percentile(depth, [2, 98])
    depth = np.clip((depth - lo) / max(1e-6, hi - lo), 0, 1)
    return depth


# ----------------------------
# Depth refinement & parallax map
# ----------------------------

def edge_preserve(depth: np.ndarray, rgb: np.ndarray, radius: int = 20, eps: float = 5e-4) -> np.ndarray:
    depth_f = depth.astype(np.float32)
    guide = np.clip(linear_to_srgb(np.clip(rgb, 0.0, 1.0)), 0.0, 1.0)
    guide_u8 = np.clip(guide * 255.0, 0, 255).astype(np.uint8)
    if _HAS_GUIDED_FILTER:
        guided = cv2.ximgproc.guidedFilter(guide_u8, depth_f, radius, eps)
        return np.clip(guided, 0.0, 1.0).astype(np.float32)

    filtered = cv2.bilateralFilter(depth_f, d=0, sigmaColor=0.06, sigmaSpace=9)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def gradient_mask(depth: np.ndarray, strength: float = 3.0) -> np.ndarray:
    lap = cv2.Laplacian(depth, cv2.CV_32F)
    mask = 1.0 - strength * np.abs(lap)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    return np.clip(mask, 0.25, 1.0)


def boundary_fade(shape: tuple[int, int], margin: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    edge = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]) / margin
    return np.clip(edge, 0.0, 1.0)


def build_parallax(depth: np.ndarray) -> np.ndarray:
    med = float(np.median(depth))
    disp = depth - med
    denom = np.percentile(np.abs(disp), 97)
    disp = np.clip(disp / max(1e-6, denom), -1.0, 1.0)
    disp = np.sign(disp) * np.power(np.abs(disp), 0.9, where=np.abs(disp) > 0, out=np.zeros_like(disp))
    disp *= boundary_fade(depth.shape, margin=max(32.0, 0.08 * max(depth.shape)))
    disp *= gradient_mask(depth, strength=2.2)
    return disp.astype(np.float32)


# ----------------------------
# Layer preparation
# ----------------------------



def _default_depth_cuts(depth: np.ndarray, layers: int) -> Sequence[float]:
    """Edge-weighted cuts: concentrate planes where depth changes."""
    if layers <= 1:
        return ()
    g = cv2.GaussianBlur(depth, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)  # large at boundaries

    hist, bins = np.histogram(depth, bins=256, range=(0.0, 1.0), weights=mag + 1e-3)
    cdf = np.cumsum(hist) / max(1e-6, np.sum(hist))

    # Bias toward near side but respect edges
    targets = np.linspace(0.85, 0.15, layers - 1, dtype=np.float32)

    cuts: list[float] = []
    for t in targets:
        i = int(np.searchsorted(cdf, t))
        i = min(max(i, 1), 255)
        cuts.append(float(bins[i]))

    # strictly decreasing & valid
    for i in range(1, len(cuts)):
        cuts[i] = min(cuts[i], cuts[i - 1] - 1e-3)
    return tuple(float(np.clip(c, 0.0, 1.0)) for c in cuts)


def soft_slices(depth: np.ndarray, cuts: Sequence[float], sharp: float = 40.0) -> list[np.ndarray]:
    if not cuts:
        return [np.ones_like(depth, dtype=np.float32)]
    boundaries = [1.0, *cuts, 0.0]

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-sharp * x))

    responses = [_sigmoid(depth - b) for b in boundaries]
    layers = [
        np.clip(responses[i + 1] - responses[i], 0.0, 1.0).astype(np.float32)
        for i in range(len(boundaries) - 1)
    ]
    stack = np.clip(np.sum(layers, axis=0), 1e-6, None)
    return [layer / stack for layer in layers]


def _prepare_alpha_map(mask: np.ndarray, erode_iter: int = 1, blur_sigma: float = 0.8) -> np.ndarray:
    mask_f = np.clip(mask, 0.0, 1.0).astype(np.float32)
    if erode_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask_f = cv2.erode(mask_f, kernel, iterations=erode_iter)
    if blur_sigma > 0.0:
        mask_f = cv2.GaussianBlur(mask_f, (0, 0), blur_sigma)
    return np.clip(mask_f, 0.0, 1.0).astype(np.float32)


def _camera_motion(t: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Looping camera offsets for pan along x/y."""
    t_arr = np.asarray(t, dtype=np.float32)
    angle = 2.0 * np.pi * t_arr
    panx = 0.45 * np.sin(angle)
    pany = 0.35 * (0.6 * np.sin(angle) + 0.4 * np.sin(2.0 * angle))
    if np.isscalar(t):  # keep scalar outputs for scalar inputs
        return float(panx), float(pany)
    return panx, pany


def _max_reveal_radius(disp: np.ndarray, base_amp: float, frames: int = 120) -> int:
    """Worst-case pixel shift magnitude across the camera path."""
    t = np.linspace(0.0, 1.0, frames, endpoint=False, dtype=np.float32)
    panx, pany = _camera_motion(t)
    # magnitude over time
    mag = np.sqrt((disp[..., None] * (base_amp * panx[None, None, :])) ** 2 +
                  (disp[..., None] * (0.75 * base_amp * pany[None, None, :])) ** 2)
    r = float(np.ceil(np.max(mag)))  # scalar is fine (safe overestimate)
    return int(np.clip(r, 3, 64))    # clamp to sane bounds


def split_layers(
    rgb: np.ndarray,
    depth: np.ndarray,
    disp: np.ndarray,
    reveal_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    near_mask = (depth > np.quantile(depth, 0.65)).astype(np.float32)
    near_mask = cv2.GaussianBlur(near_mask, (0, 0), 6.0)
    near_mask = np.clip(near_mask, 0, 1)
    far_mask = 1.0 - near_mask

    # One big inpaint plate for the worst-case reveal
    mask_u8 = (near_mask > 0.25).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reveal_radius + 1, 2 * reveal_radius + 1))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)

    rgb_srgb = np.clip(linear_to_srgb(np.clip(rgb, 0.0, 1.0)), 0.0, 1.0)
    bgr = cv2.cvtColor((rgb_srgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    method = cv2.INPAINT_NS if reveal_radius >= 24 else cv2.INPAINT_TELEA
    inpainted = cv2.inpaint(bgr, dilated, 7, method)
    bg = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    bg = srgb_to_linear(bg)  # keep linear-light internally

    return near_mask, far_mask, bg


# ----------------------------
# Warping helpers
# ----------------------------

class CpuWarpContext:
    def __init__(self, shape: tuple[int, int]):
        h, w = shape
        self.xx, self.yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    def warp(self, img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        map_x = (self.xx + dx).astype(np.float32)
        map_y = (self.yy + dy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)


class TorchWarpContext:
    def __init__(self, shape: tuple[int, int], device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.h, self.w = shape
        self.scale_x = 2.0 / max(self.w - 1, 1)
        self.scale_y = 2.0 / max(self.h - 1, 1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.h, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, self.w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)

    def warp(self, img: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        flow_x = dx.squeeze(1) * self.scale_x
        flow_y = dy.squeeze(1) * self.scale_y
        grid = self.base_grid.to(img.dtype) + torch.stack((flow_x, flow_y), dim=-1)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


# ----------------------------
# Rendering
# ----------------------------



def generate_frames(
    rgb: np.ndarray,
    depth: np.ndarray,
    disp: np.ndarray,
    seconds: float,
    fps: int,
    parallax_px: float,
    layers: int,
    use_cuda_warp: bool,
    device_cfg: DeviceConfig,
) -> list[np.ndarray]:
    h, w = depth.shape
    frames = max(1, int(round(seconds * fps)))

    base_amp = parallax_px / 1080.0 * h
    reveal_radius = _max_reveal_radius(disp, base_amp)
    near_mask, far_mask, bg = split_layers(rgb, depth, disp, reveal_radius)
    layers = max(2, layers)

    if layers <= 2:
        weights = [np.clip(near_mask, 0.0, 1.0), np.clip(far_mask, 0.0, 1.0)]
    else:
        cuts = _default_depth_cuts(depth, layers)
        weights = soft_slices(depth, cuts)

    colors: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for idx, weight in enumerate(weights):
        if idx == layers - 1:
            color = bg
        else:
            if layers == 2:
                color = rgb
            else:
                blend = (idx / max(layers - 1, 1)) ** 1.5
                color = rgb * (1.0 - blend) + bg * blend
        colors.append(np.clip(color, 0.0, 1.0).astype(np.float32))
        erode_iter = max(1, reveal_radius // 4) if idx == 0 else 0
        alphas.append(_prepare_alpha_map(weight, erode_iter=erode_iter, blur_sigma=0.8))

    alphas[-1][:] = 1.0

    parallax_scales = np.linspace(1.2, 0.85, layers, dtype=np.float32)
    use_cuda = use_cuda_warp and device_cfg.torch_device.type == "cuda"

    outputs: list[np.ndarray] = []
    if use_cuda:
        dtype = torch.float16 if device_cfg.use_half else torch.float32
        warp_ctx = TorchWarpContext((h, w), device_cfg.torch_device, dtype)
        disp_t = torch.from_numpy(disp).to(device_cfg.torch_device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        colors_t = []
        for col in colors:
            col_t = torch.from_numpy(col.transpose(2, 0, 1)).unsqueeze(0)
            col_t = col_t.to(device_cfg.torch_device, dtype=dtype)
            colors_t.append(col_t.contiguous(memory_format=torch.channels_last))
        alphas_t = [
            torch.from_numpy(alpha).to(device_cfg.torch_device, dtype=dtype).unsqueeze(0).unsqueeze(0).contiguous()
            for alpha in alphas
        ]

        acc = torch.zeros_like(colors_t[0])
        acc_a = torch.zeros_like(alphas_t[0])
        for i in range(frames):
            t = i / max(1, frames)
            panx, pany = _camera_motion(t)
            dx_base = disp_t * (base_amp * panx)
            dy_base = disp_t * (0.75 * base_amp * pany)

            acc.zero_()
            acc_a.zero_()
            for idx in reversed(range(layers)):
                scale = float(parallax_scales[idx])
                dx_layer = dx_base * scale
                dy_layer = dy_base * scale
                warped_alpha = torch.clamp(warp_ctx.warp(alphas_t[idx], dx_layer, dy_layer), 0.0, 1.0)
                warped_color = torch.clamp(warp_ctx.warp(colors_t[idx], dx_layer, dy_layer), 0.0, 1.0)
                inv_alpha = 1.0 - warped_alpha
                alpha_exp = warped_alpha.expand_as(acc)
                inv_alpha_exp = inv_alpha.expand_as(acc)
                acc.mul_(inv_alpha_exp).addcmul_(warped_color, alpha_exp)
                acc_a.mul_(inv_alpha).add_(warped_alpha)

            frame = acc.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).float().cpu().numpy()
            outputs.append(frame.astype(np.float32))
    else:
        warp_ctx = CpuWarpContext((h, w))
        disp_np = disp.astype(np.float32)
        for i in range(frames):
            t = i / max(1, frames)
            panx, pany = _camera_motion(t)
            dx_base = base_amp * panx * disp_np
            dy_base = 0.75 * base_amp * pany * disp_np

            acc = np.zeros_like(rgb, dtype=np.float32)
            acc_a = np.zeros((h, w), dtype=np.float32)
            for idx in reversed(range(layers)):
                scale = float(parallax_scales[idx])
                dx_layer = dx_base * scale
                dy_layer = dy_base * scale
                warped_alpha = np.clip(warp_ctx.warp(alphas[idx], dx_layer, dy_layer), 0.0, 1.0)
                warped_color = np.clip(warp_ctx.warp(colors[idx], dx_layer, dy_layer), 0.0, 1.0)
                acc = warped_color * warped_alpha[..., None] + acc * (1.0 - warped_alpha[..., None])
                acc_a = warped_alpha + acc_a * (1.0 - warped_alpha)

            outputs.append(np.clip(acc, 0.0, 1.0))

    return outputs




def _encode_hevc_nvenc(src: Path, dst: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "hevc_nvenc",
        "-pix_fmt",
        "p010le",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-preset",
        "p5",
        "-profile:v",
        "main10",
        str(dst),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            stderr = exc.stderr.decode("utf-8", errors="ignore")
        else:
            stderr = str(exc)
        print(f"HEVC NVENC encode error: {stderr.strip()[:240]}")
        return False
    return True


def write_video(frames: list[np.ndarray], path: Path, fps: int, hevc_nvenc: bool = True) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    tmp_path = path.with_name(f"{path.stem}_mp4v.mp4") if hevc_nvenc else path
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {tmp_path}")
    for frame in frames:
        # convert back to sRGB just before encoding
        out = np.clip(linear_to_srgb(np.clip(frame, 0.0, 1.0)), 0.0, 1.0)
        bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

    if hevc_nvenc:
        success = _encode_hevc_nvenc(tmp_path, path)
        if success:
            print(f"Re-encoded {path.name} with HEVC NVENC (10-bit).")
            tmp_path.unlink(missing_ok=True)
        else:
            print("HEVC NVENC encode failed; keeping mp4v output.")
            if tmp_path != path:
                tmp_path.replace(path)


# ----------------------------
# CLI
# ----------------------------

def run(args: argparse.Namespace) -> None:
    device_cfg = pick_device(args.device)
    use_cuda = args.cuda_warp and device_cfg.torch_device.type == "cuda"
    print(f"Using device: {device_cfg.torch_device}" + (" (fp16)" if device_cfg.use_half else ""))

    start_t = time.time()
    rgb_lin = load_image(Path(args.input), args.max_dim)
    load_t = time.time() - start_t
    print(f"Loaded image {rgb_lin.shape[1]}x{rgb_lin.shape[0]} in {load_t:.2f}s")

    depth_start = time.time()
    # Depth in FP32 for quality; warps remain in fp16 if CUDA
    depth_device = DeviceConfig(device_cfg.torch_device, use_half=False)
    depth = estimate_depth(
        rgb_lin,
        depth_device,
        model_name=args.depth_model,
        depth_max_dim=args.depth_max_dim,
    )
    depth = edge_preserve(depth, rgb_lin)
    depth_t = time.time() - depth_start
    print(f"Depth ({args.depth_model}) estimated in {depth_t:.2f}s at max side {args.depth_max_dim}px")

    render_start = time.time()

    rgb_render = rgb_lin
    if args.render_scale > 1.0:
        h, w = rgb_render.shape[:2]
        W = int(round(w * args.render_scale))
        H = int(round(h * args.render_scale))
        rgb_render = cv2.resize(rgb_render, (W, H), cv2.INTER_LANCZOS4)
        depth = cv2.resize(depth, (W, H), cv2.INTER_CUBIC)

    disp = build_parallax(depth)
    frames = generate_frames(
        rgb_render,
        depth,
        disp,
        args.seconds,
        args.fps,
        args.parallax_px,
        args.layers,
        args.cuda_warp,
        device_cfg,
    )
    render_t = time.time() - render_start
    warp_label = "cuda" if use_cuda else "cpu"
    print(f"Generated {len(frames)} frames with {args.layers} layers using {warp_label} warp in {render_t:.2f}s")

    write_video(frames, Path(args.output), args.fps, hevc_nvenc=args.hevc_nvenc)
    total = time.time() - start_t
    print(f"Saved {len(frames)} frames to {args.output} (total {total:.2f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="High-quality parallax animation")
    ap.add_argument("input", type=str, help="Path to input image")
    ap.add_argument("output", type=str, help="Path to output MP4 video")
    ap.add_argument("--seconds", type=float, default=6.0, help="Clip length in seconds")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second")
    ap.add_argument("--max-dim", type=int, default=2048, help="Resize largest image dimension to this (keeps aspect)")
    ap.add_argument("--parallax-px", type=float, default=9.0, help="Parallax amplitude at 1080p in pixels")
    ap.add_argument("--depth-model", type=str, default="depth-anything/Depth-Anything-V2-Large-hf",
                    help="Hugging Face depth model identifier")
    ap.add_argument("--depth-max-dim", type=int, default=1024,
                    help="Resize largest image dimension for depth estimation")
    ap.add_argument("--render-scale", type=float, default=1.0,
                    help="Supersample render scale (e.g., 1.5 or 2.0)")
    ap.add_argument("--layers", type=int, default=4, help="Number of soft MPI layers (>=2)")
    ap.add_argument("--cuda-warp", action="store_true", default=True,
                    help="Use torch.grid_sample for warps on CUDA (default)")
    ap.add_argument("--cpu-warp", dest="cuda_warp", action="store_false",
                    help="Force CPU remap warping instead of CUDA")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Torch device preference")
    ap.add_argument("--no-hevc", dest="hevc_nvenc", action="store_false",
                    help="Skip HEVC 10-bit NVENC re-encode; keep mp4v output")
    ap.set_defaults(hevc_nvenc=True)
    args = ap.parse_args()

    run(args)


if __name__ == "__main__":
    main()
