#!/usr/bin/env python3
"""Generate an animated clip with FramePack Studio using an optional reference image."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path, PureWindowsPath

import httpx
from gradio_client import Client, handle_file

FRAMEPACK_URL = "http://127.0.0.1:7860"
POLL_INTERVAL_SECONDS = 5
MIN_DURATION_SECONDS = 0.1
MAX_DURATION_SECONDS = 120.0


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for the animation helper."""

    parser = argparse.ArgumentParser(
        description="Render a FramePack animation using an optional reference image."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to the source image; omit to generate from noise only",
    )
    parser.add_argument("prompt", help="Prompt that describes the desired animation")
    parser.add_argument(
        "duration",
        type=float,
        help="Desired clip length in seconds (FramePack range is roughly 0.1-120)",
    )
    parser.add_argument(
        "--model",
        choices=("Original", "F1"),
        default="Original",
        help="FramePack model variant to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output filename (defaults to <image>_animation.mp4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print job status updates while waiting for completion",
    )
    return parser.parse_args()


def submit_job(
    client: Client,
    image_path: Path | None,
    prompt: str,
    duration: float,
    model: str,
    *,
    end_frame: Path | None = None,
    end_frame_influence: float = 1.0,
    negative_prompt: str = "",
    seed: int = 2500,
):
    """Kick off an animation job and return FramePack's job identifier."""

    start_frame = handle_file(str(image_path)) if image_path is not None else None
    end_frame_input = handle_file(str(end_frame)) if end_frame is not None else None

    result = client.predict(
        model,  # Generation Type
        start_frame,  # Start Frame
        None,  # Video Input (unused)
        end_frame_input,  # End Frame
        end_frame_influence,  # End Frame Influence
        prompt,
        negative_prompt,
        seed,
        False,  # Randomize
        duration,  # Video length (seconds)
        9,  # Latent window size
        25,  # Steps
        1.0,  # CFG Scale
        10.0,  # Distilled CFG Scale
        0.0,  # CFG Re-Scale
        "MagCache",  # Caching strategy
        25,  # TeaCache steps
        0.15,  # TeaCache rel_l1_thresh
        0.1,  # MagCache Threshold
        2,  # MagCache Max Consecutive Skips
        0.25,  # MagCache Retention Ratio
        4,  # Sections to blend
        "Noise",  # Latent image seed
        True,  # Clean up video files
        [],  # LoRAs
        640,  # Width
        640,  # Height
        False,  # Combine with source video
        5,  # Context frames
        api_name="/handle_start_button",
    )
    job_id = result[1]
    if not job_id:
        raise RuntimeError("FramePack Studio did not return a job id")
    return job_id


def _extract_video_path(video_info) -> str | None:
    """Return the first file path contained in the monitor response payload."""

    def _extract(candidate) -> str | None:
        """Recursively walk nested results to locate file-like entries."""

        if candidate is None:
            return None
        if isinstance(candidate, (str, Path)):
            return str(candidate)
        if isinstance(candidate, dict):
            for key in ("video", "path", "name", "value"):
                if key in candidate:
                    path = _extract(candidate[key])
                    if path:
                        return path
            return None
        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                path = _extract(item)
                if path:
                    return path
        return None

    return _extract(video_info)


def wait_for_completion(client: Client, job_id: str, *, verbose: bool = False) -> str:
    """Poll FramePack until the job completes and return the reported file path."""

    while True:
        poll = client.predict(job_id, api_name="/monitor_job")
        status_raw = str(poll[3]).strip()
        status = status_raw.lower()
        # FramePack reports `Completed` or `JobStatus.COMPLETED`; checking the
        # substring keeps this resilient to minor copy changes.
        if "completed" in status:
            video_info = poll[0]
            video_path = _extract_video_path(video_info)
            if video_path:
                if verbose:
                    print(f"Job {job_id} completed")
                return video_path
            raise RuntimeError(
                "FramePack job completed but response did not contain a video path: "
                f"{video_info!r}"
            )
        if "failed" in status or "error" in status:
            raise RuntimeError(f"FramePack job {job_id} failed: {poll}")
        if verbose:
            print(f"Job {job_id} status: {status_raw}")
        time.sleep(POLL_INTERVAL_SECONDS)


def _sanitize_filename(name: str) -> str:
    """Remove characters that are invalid on Windows filesystems."""

    invalid = set('<>:"/\\|?*')
    cleaned = "".join("_" if ch in invalid else ch for ch in name)
    return cleaned or "framepack_video.mp4"


def _ensure_local_video(
    client: Client,
    video_path: str,
    download_dir: Path,
    job_id: str,
    *,
    auto_download: bool,
) -> Path:
    """Return a local path to the rendered video, downloading if the client didn't."""

    # FramePack may return an absolute path when the client is running on Linux.
    candidate = Path(video_path)
    if candidate.exists():
        return candidate

    # Sometimes Gradio returns a relative path when download_files is enabled.
    # When `download_files` is True, gradio writes into the provided directory and
    # returns the same relative path. Handle that case directly.
    rel_candidate = (download_dir / video_path).resolve()
    if rel_candidate.exists():
        return rel_candidate

    # At this point gradio either failed to download the file, or we deliberately
    # disabled the behaviour (e.g. on Windows). If download was expected and the
    # file is missing, fail early with a clearer error message.
    if auto_download:
        raise FileNotFoundError(
            f"Expected downloaded video at {video_path!r} but it was not found"
        )

    # Manual download from FramePack server.
    remote_name = video_path
    # Gradio may embed a Windows path inside the filename; preserve only the
    # basename so the resulting filename is valid on both platforms.
    if "\\" in remote_name or ":" in remote_name:
        remote_basename = PureWindowsPath(remote_name).name
    else:
        remote_basename = Path(remote_name).name or remote_name

    filename = _sanitize_filename(f"{job_id}_{remote_basename}")
    destination = download_dir / filename

    # FramePack serves files from `<client.src>/file=<remote path>`; URL-encode the
    # remote segment so Windows drive letters and backslashes are accepted.
    url_base = client.src if client.src.endswith("/") else client.src + "/"
    encoded_remote = urllib.parse.quote(remote_name, safe="/")
    file_url = f"{url_base}file={encoded_remote}"

    download_dir.mkdir(parents=True, exist_ok=True)

    with httpx.stream(
        "GET",
        file_url,
        headers=client.headers,
        cookies=client.cookies,
        verify=client.ssl_verify,
        follow_redirects=True,
        **client.httpx_kwargs,
    ) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    return destination.resolve()


def main() -> int:
    args = parse_args()
    image_path: Path | None = args.image
    if image_path is not None:
        image_path = image_path.expanduser().resolve()
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return 1

    if not (MIN_DURATION_SECONDS <= args.duration <= MAX_DURATION_SECONDS):
        print(
            "Duration must be between "
            f"{MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS} seconds",
            file=sys.stderr,
        )
        return 2

    out_path = args.output
    if out_path is None:
        if image_path is not None:
            out_path = image_path.with_name(image_path.stem + "_animation.mp4")
        else:
            out_path = Path("framepack_animation.mp4")
    out_path = out_path.expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="framepack_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        # Gradio's download helper struggles with Windows paths inside the WSL
        # mount, so let it write files automatically only on POSIX systems.
        auto_download = os.name != "nt"
        download_setting: str | Path | bool = temp_dir if auto_download else False

        client = Client(FRAMEPACK_URL, download_files=download_setting)

        video_local: Path | None = None
        try:
            job_id = submit_job(
                client,
                image_path,
                args.prompt,
                args.duration,
                args.model,
            )
            video_path = wait_for_completion(
                client, job_id, verbose=args.verbose
            )
            video_local = _ensure_local_video(
                client,
                video_path,
                temp_dir,
                job_id,
                auto_download=auto_download,
            )
        finally:
            # Ensure background threads shut down so the script can exit cleanly.
            close_method = getattr(client, "close", None)
            if callable(close_method):
                close_method()
            executor = getattr(client, "executor", None)
            if executor is not None:
                # Cancel pending work so lingering HTTP futures do not keep the
                # interpreter alive.
                executor.shutdown(wait=True, cancel_futures=True)

        if video_local is None:
            raise RuntimeError("FramePack job did not produce a downloadable video")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video_local, out_path)
        print(f"Saved video to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
