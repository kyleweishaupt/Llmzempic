"""Example images: content-addressed files in <project>/images/, snapshotted into each run's images/."""

import hashlib
import os
import re
import shutil
import time
from pathlib import Path

MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_NAME = re.compile(r"^[0-9a-f]{24}\.(jpg|png|webp|gif)$")
MEDIA_TYPES = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
ORPHAN_SECONDS = 24 * 3600


class ImageError(Exception):
	pass


def image_type(data):
	if data[:3] == b"\xff\xd8\xff":
		return "jpg"
	if data[:8] == b"\x89PNG\r\n\x1a\n":
		return "png"
	if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
		return "webp"
	if data[:6] in (b"GIF87a", b"GIF89a"):
		return "gif"
	return None


def save_image(folder, data):
	"""Store image bytes under their content hash and return the file name (same bytes, same name)."""
	if not data:
		raise ImageError("The image is empty.")
	if len(data) > MAX_IMAGE_BYTES:
		raise ImageError(f"Images can be at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
	ext = image_type(data)
	if not ext:
		raise ImageError("Only JPEG, PNG, WebP and GIF images are supported.")
	name = f"{hashlib.sha256(data).hexdigest()[:24]}.{ext}"
	path = Path(folder) / name
	if path.exists():
		os.utime(path)  # a fresh upload of a known image must survive orphan cleanup too
		return name
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(f"{name}.{os.getpid()}.tmp")
	tmp.write_bytes(data)
	os.replace(tmp, path)
	return name


def image_path(folder, name):
	if not IMAGE_NAME.match(name or ""):
		raise KeyError(name)
	path = Path(folder) / name
	if not path.is_file():
		raise KeyError(name)
	return path


def media_type(name):
	return MEDIA_TYPES[name.rsplit(".", 1)[-1]]


def link_images(names, source, dest):
	"""Snapshot images into a run; hard links when the filesystem allows (images never change)."""
	dest = Path(dest)
	for name in set(names):
		src = image_path(source, name)
		target = dest / name
		if target.exists():
			continue
		dest.mkdir(parents=True, exist_ok=True)
		try:
			os.link(src, target)
		except OSError:
			shutil.copy2(src, target)


def remove_orphans(folder, referenced, now=None):
	"""Delete images no example uses. Recent uploads stay: their example may not be saved yet."""
	folder = Path(folder)
	if not folder.is_dir():
		return
	cutoff = (now or time.time()) - ORPHAN_SECONDS
	for path in folder.iterdir():
		if IMAGE_NAME.match(path.name) and path.name not in referenced and path.stat().st_mtime < cutoff:
			path.unlink(missing_ok=True)
