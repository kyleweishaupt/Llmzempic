"""Detect GPUs, memory and optional libraries without importing torch into the UI process.

Importing torch (and touching CUDA) in the server would hold GPU memory that training needs,
so the torch probe runs once in a short-lived subprocess.
"""

import ctypes
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys

_PROBE = r"""
import json
info = {"torch_ok": False}
try:
	import torch
	info["torch_ok"] = True
	info["cuda"] = torch.cuda.is_available()
	info["cuda_version"] = torch.version.cuda
	info["rocm"] = getattr(torch.version, "hip", None)
	info["gpus"] = []
	if info["cuda"]:
		for i in range(torch.cuda.device_count()):
			p = torch.cuda.get_device_properties(i)
			info["gpus"].append({"name": p.name, "vram_gb": round(p.total_memory / 1024**3, 1)})
		info["bf16"] = torch.cuda.is_bf16_supported()
	mps = getattr(torch.backends, "mps", None)
	info["mps"] = bool(mps and mps.is_available())
except Exception as e:
	info["error"] = str(e)
print(json.dumps(info))
"""

_TRAIN_PACKAGES = ["torch", "transformers", "peft", "accelerate", "safetensors"]
# Image projects also decode images (Pillow); Qwen-VL processors need torchvision.
_IMAGE_PACKAGES = ["pillow", "torchvision"]


def _version(dist):
	try:
		return importlib.metadata.version(dist)
	except importlib.metadata.PackageNotFoundError:
		return None


def ram_gb():
	try:
		if os.name == "nt":
			class MemoryStatus(ctypes.Structure):
				_fields_ = [
					("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
					("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
					("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
					("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
					("ullAvailExtendedVirtual", ctypes.c_ulonglong),
				]
			status = MemoryStatus()
			status.dwLength = ctypes.sizeof(MemoryStatus)
			ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
			return round(status.ullTotalPhys / 1024**3, 1)
		with open("/proc/meminfo") as f:
			for line in f:
				if line.startswith("MemTotal:"):
					return round(int(line.split()[1]) / 1024**2, 1)
	except Exception:
		pass
	try:
		return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
	except (AttributeError, ValueError, OSError):
		return None


def probe_torch(timeout=120):
	if importlib.util.find_spec("torch") is None:
		return {"torch_ok": False, "error": "PyTorch is not installed"}
	try:
		out = subprocess.run(
			[sys.executable, "-c", _PROBE],
			capture_output=True, text=True, timeout=timeout,
		)
		return json.loads(out.stdout.strip().splitlines()[-1])
	except Exception as e:
		return {"torch_ok": False, "error": f"PyTorch probe failed: {e}"}


def detect():
	torch_info = probe_torch()
	packages = {p: _version(p) for p in _TRAIN_PACKAGES + _IMAGE_PACKAGES + ["bitsandbytes", "unsloth"]}
	missing = [p for p in _TRAIN_PACKAGES if not packages[p]]
	missing_image = [p for p in _IMAGE_PACKAGES if not packages[p]]
	gpus = torch_info.get("gpus") or []
	if torch_info.get("cuda") and gpus:
		device = "cuda"
	elif torch_info.get("mps"):
		device = "mps"
	else:
		device = "cpu"
	return {
		"python": platform.python_version(),
		"os": f"{platform.system()} {platform.release()}",
		"machine": platform.machine(),
		"cpu_count": os.cpu_count(),
		"ram_gb": ram_gb(),
		"device": device,
		"gpus": gpus,
		"vram_gb": max((g["vram_gb"] for g in gpus), default=0),
		"bf16": bool(torch_info.get("bf16")),
		"torch": torch_info,
		"packages": packages,
		"missing_train_packages": missing,
		"can_train": torch_info.get("torch_ok", False) and not missing,
		"missing_image_packages": missing_image,
		"can_train_images": torch_info.get("torch_ok", False) and not missing and not missing_image,
		"can_4bit": device == "cuda" and bool(packages["bitsandbytes"]),
		"can_unsloth": device == "cuda" and bool(packages["unsloth"]),
		"ollama": shutil.which("ollama"),
	}


def estimate_memory_gb(params_b, four_bit):
	"""Rough peak memory for LoRA training with short sequences and gradient checkpointing."""
	if four_bit:
		return round(params_b * 0.7 + 2.0, 1)
	return round(params_b * 2.3 + 2.0, 1)


def model_fit(params_b, hw):
	"""Return (fit, detail) where fit is good | tight | slow | too-big."""
	if hw["device"] == "cuda":
		vram = hw["vram_gb"]
		full = estimate_memory_gb(params_b, False)
		quant = estimate_memory_gb(params_b, True)
		if full <= vram * 0.85:
			return "good", f"~{full} GB VRAM"
		if hw["can_4bit"] and quant <= vram * 0.9:
			return "good", f"~{quant} GB VRAM (4-bit)"
		if quant <= vram:
			return "tight", f"~{quant} GB VRAM (needs 4-bit)"
		return "too-big", f"needs ~{quant} GB VRAM"
	ram = hw.get("ram_gb") or 8
	need = round(params_b * 4.5 + 2, 1)
	if need > ram:
		return "too-big", f"needs ~{need} GB RAM on CPU"
	if params_b <= 0.6:
		return "slow", "CPU training: minutes to hours"
	return "slow", "CPU training is very slow; use a GPU"
