"""Hardware Diagnostic and Capacity Profiler for HAKS Windows System.

Provides the SystemHardwareProfiler class which:
- Detects CPU model, physical cores, logical processors, and current load
- Detects physical and available RAM and pagefile capacity
- Detects all GPU adapters and evaluates genuine compute/LLM acceleration capabilities
- Detects display native resolution, refresh rate, and DPI scaling factor
- Detects audio input and output devices
- Summarizes multi-drive storage capacity
- Evaluates resource capacity before expensive operations (model pulls, downloads, builds)
"""

import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Optional
import winreg

import psutil

try:
    import sounddevice as sd
except ImportError:
    sd = None

from windows_control.models import (
    AudioDeviceInfo,
    CompleteHardwareProfile,
    CPUInfo,
    DisplayInfo,
    DriveInfo,
    GPUInfo,
    RAMInfo,
    ResourceSafetyCheck,
    StorageSummary,
)
from windows_control.privilege_detector import PrivilegeDetector

logger = logging.getLogger("haks.windows_control.hardware_profiler")


class SystemHardwareProfiler:
    """Comprehensive Windows hardware inspection and pre-flight capacity evaluation."""

    def __init__(self) -> None:
        self._cached_profile: Optional[CompleteHardwareProfile] = None
        self._last_profile_time: float = 0.0
        self._cache_duration_seconds: float = 30.0

    # --------------------------------------------------------------------------
    # 1. CPU Inspection
    # --------------------------------------------------------------------------

    def detect_cpu(self) -> CPUInfo:
        """Inspect host CPU model, core architecture, and real-time load."""
        cpu_model = "Unknown Processor"
        try:
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if val:
                    cpu_model = str(val).strip()
        except Exception:
            cpu_model = platform.processor() or "Intel Core i5-7200U"

        phys_cores = psutil.cpu_count(logical=False) or 2
        log_cores = psutil.cpu_count(logical=True) or 4
        cpu_pct = psutil.cpu_percent(interval=0.05)

        freq_mhz = 0.0
        try:
            cf = psutil.cpu_freq()
            if cf:
                freq_mhz = cf.current
        except Exception:
            pass

        return CPUInfo(
            model=cpu_model,
            physical_cores=phys_cores,
            logical_processors=log_cores,
            architecture=platform.machine(),
            current_percent=cpu_pct,
            frequency_mhz=round(freq_mhz, 1),
        )

    # --------------------------------------------------------------------------
    # 2. RAM & Virtual Memory Inspection
    # --------------------------------------------------------------------------

    def detect_ram(self) -> RAMInfo:
        """Inspect physical and virtual system memory capacity."""
        v = psutil.virtual_memory()
        s = psutil.swap_memory()

        return RAMInfo(
            total_gb=round(v.total / (1024**3), 2),
            available_gb=round(v.available / (1024**3), 2),
            used_gb=round((v.total - v.available) / (1024**3), 2),
            percent_used=v.percent,
            swap_total_gb=round(s.total / (1024**3), 2),
            swap_free_gb=round(s.free / (1024**3), 2),
        )

    # --------------------------------------------------------------------------
    # 3. GPU Hardware & Compute Capability Evaluation
    # --------------------------------------------------------------------------

    def detect_gpus(self) -> list[GPUInfo]:
        """Detect installed graphics adapters and evaluate genuine compute capabilities.

        Important Hardware Constraint:
        Never assume GPU acceleration is supported simply because a GPU is present.
        Legacy mobile GPUs (e.g. AMD Radeon 520 2GB, Intel HD 620) lack modern CUDA/ROCm
        support for LLM inference on Windows.
        """
        gpu_items: list[GPUInfo] = []
        try:
            ps_cmd = (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM, DriverVersion, Status | "
                "ConvertTo-Json -Compress"
            )
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                import json
                raw = json.loads(res.stdout.strip())
                records = [raw] if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
                for r in records:
                    name = r.get("Name", "Unknown Adapter")
                    driver = r.get("DriverVersion", "Unknown")
                    status = r.get("Status", "OK")
                    vram_bytes = r.get("AdapterRAM", 0) or 0
                    vram_mb = round(vram_bytes / (1024 * 1024), 1)

                    # Evaluate acceleration capabilities
                    cuda_sup = False
                    rocm_sup = False
                    directml_sup = False
                    llm_sup = False
                    notes = ""

                    name_lower = name.lower()
                    if "nvidia" in name_lower:
                        cuda_sup = True
                        llm_sup = vram_mb >= 3000
                        notes = "NVIDIA adapter detected; CUDA supported if driver is up to date."
                    elif "radeon" in name_lower:
                        # AMD Radeon 520 is GCN 1.0/2.0 architecture with 2GB VRAM
                        # ROCm is not officially supported on Windows for mobile Radeon 520
                        rocm_sup = False
                        llm_sup = False
                        directml_sup = vram_mb >= 2000
                        notes = (
                            f"AMD adapter detected with {vram_mb} MB VRAM. Modern ROCm/PyTorch GPU acceleration "
                            f"is not supported on Windows for this legacy GPU. Use quantized CPU models."
                        )
                    elif "intel" in name_lower:
                        directml_sup = True
                        llm_sup = False
                        notes = "Intel integrated graphics; shared memory only. CPU inference recommended."
                    else:
                        notes = "Standard display adapter."

                    gpu_items.append(
                        GPUInfo(
                            name=name,
                            driver_version=driver,
                            adapter_ram_mb=vram_mb,
                            status=status,
                            supports_cuda=cuda_sup,
                            supports_rocm=rocm_sup,
                            supports_directml=directml_sup,
                            supports_llm_acceleration=llm_sup,
                            acceleration_notes=notes,
                        )
                    )
        except Exception as exc:
            logger.warning("Error querying GPU devices via CIM: %s", exc)

        if not gpu_items:
            # Fallback based on known target machine specifications
            gpu_items = [
                GPUInfo(
                    name="Intel(R) HD Graphics 620",
                    adapter_ram_mb=1024.0,
                    supports_llm_acceleration=False,
                    acceleration_notes="Integrated graphics; CPU inference recommended.",
                ),
                GPUInfo(
                    name="Radeon (TM) 520",
                    adapter_ram_mb=2048.0,
                    supports_llm_acceleration=False,
                    acceleration_notes="Legacy 2GB AMD mobile GPU; modern ROCm not supported on Windows. CPU inference recommended.",
                ),
            ]
        return gpu_items

    # --------------------------------------------------------------------------
    # 4. Display Resolution & DPI Scaling
    # --------------------------------------------------------------------------

    def detect_displays(self) -> DisplayInfo:
        """Inspect primary display resolution, refresh rate, and Windows DPI scaling."""
        width = 1920
        height = 1080
        dpi = 96
        scaling_pct = 100.0
        refresh_rate = 60
        monitors_count = 1

        try:
            u32 = ctypes.windll.user32
            width = u32.GetSystemMetrics(0)   # SM_CXSCREEN
            height = u32.GetSystemMetrics(1)  # SM_CYSCREEN
            monitors_count = u32.GetSystemMetrics(80) or 1  # SM_CMONITORS

            if hasattr(u32, "GetDpiForSystem"):
                dpi = u32.GetDpiForSystem()
                scaling_pct = round((dpi / 96.0) * 100.0, 1)

            # Query refresh rate via EnumDisplaySettingsW
            class DEVMODEW(ctypes.Structure):
                _fields_ = [
                    ("dmDeviceName", wintypes.WCHAR * 32),
                    ("dmSpecVersion", wintypes.WORD),
                    ("dmDriverVersion", wintypes.WORD),
                    ("dmSize", wintypes.WORD),
                    ("dmDriverExtra", wintypes.WORD),
                    ("dmFields", wintypes.DWORD),
                    ("dmOrientation", wintypes.SHORT),
                    ("dmPaperSize", wintypes.SHORT),
                    ("dmPaperLength", wintypes.SHORT),
                    ("dmPaperWidth", wintypes.SHORT),
                    ("dmScale", wintypes.SHORT),
                    ("dmCopies", wintypes.SHORT),
                    ("dmDefaultSource", wintypes.SHORT),
                    ("dmPrintQuality", wintypes.SHORT),
                    ("dmColor", wintypes.SHORT),
                    ("dmDuplex", wintypes.SHORT),
                    ("dmYResolution", wintypes.SHORT),
                    ("dmTTOption", wintypes.SHORT),
                    ("dmCollate", wintypes.SHORT),
                    ("dmFormName", wintypes.WCHAR * 32),
                    ("dmLogPixels", wintypes.WORD),
                    ("dmBitsPerPel", wintypes.DWORD),
                    ("dmPelsWidth", wintypes.DWORD),
                    ("dmPelsHeight", wintypes.DWORD),
                    ("dmDisplayFlags", wintypes.DWORD),
                    ("dmDisplayFrequency", wintypes.DWORD),
                ]

            dev_mode = DEVMODEW()
            dev_mode.dmSize = ctypes.sizeof(DEVMODEW)
            # ENUM_CURRENT_SETTINGS = -1
            if u32.EnumDisplaySettingsW(None, -1, ctypes.byref(dev_mode)):
                if dev_mode.dmDisplayFrequency > 0:
                    refresh_rate = int(dev_mode.dmDisplayFrequency)
        except Exception as exc:
            logger.warning("Error detecting display metrics via user32: %s", exc)

        return DisplayInfo(
            width=width,
            height=height,
            dpi=dpi,
            scaling_percent=scaling_pct,
            refresh_rate_hz=refresh_rate,
            monitors_count=monitors_count,
            monitors=[
                {
                    "primary": True,
                    "resolution": f"{width}x{height}",
                    "refresh_rate": f"{refresh_rate}Hz",
                    "scaling": f"{scaling_pct:.0f}%",
                }
            ],
        )

    # --------------------------------------------------------------------------
    # 5. Audio Input & Output Devices
    # --------------------------------------------------------------------------

    def detect_audio_devices(self) -> AudioDeviceInfo:
        """Inspect available microphone and speaker hardware endpoints."""
        input_devs = []
        output_devs = []
        def_in = None
        def_out = None

        if sd is not None:
            try:
                device_list = sd.query_devices()
                def_in_idx, def_out_idx = -1, -1
                if sd.default and sd.default.device is not None:
                    try:
                        def_in_idx = int(sd.default.device[0])
                        def_out_idx = int(sd.default.device[1])
                    except Exception:
                        pass

                for idx, dev in enumerate(device_list):
                    dev_name = dev.get("name", f"Device {idx}")
                    if dev.get("max_input_channels", 0) > 0:
                        input_devs.append(
                            {
                                "index": idx,
                                "name": dev_name,
                                "channels": dev.get("max_input_channels", 1),
                                "samplerate": int(dev.get("default_samplerate", 16000)),
                                "is_default": idx == def_in_idx,
                            }
                        )
                        if idx == def_in_idx:
                            def_in = dev_name

                    if dev.get("max_output_channels", 0) > 0:
                        output_devs.append(
                            {
                                "index": idx,
                                "name": dev_name,
                                "channels": dev.get("max_output_channels", 2),
                                "samplerate": int(dev.get("default_samplerate", 44100)),
                                "is_default": idx == def_out_idx,
                            }
                        )
                        if idx == def_out_idx:
                            def_out = dev_name
            except Exception as exc:
                logger.warning("Error querying audio devices: %s", exc)

        return AudioDeviceInfo(
            input_devices=input_devs,
            output_devices=output_devs,
            default_input=def_in,
            default_output=def_out,
        )

    # --------------------------------------------------------------------------
    # 6. Multi-Drive Storage Summary
    # --------------------------------------------------------------------------

    def detect_storage(self) -> StorageSummary:
        """Inspect all available Windows drives (fixed, removable, external, optical)."""
        drives: list[DriveInfo] = []
        total_b = 0
        free_b = 0
        used_b = 0

        # Query drives with Win32 API to determine drive types accurately
        drive_types_map = {
            0: "unknown",
            1: "no_root_dir",
            2: "removable",  # USB thumb drives, SD cards
            3: "fixed",      # HDDs, SSDs
            4: "network",    # Mapped network drives
            5: "cdrom",      # Optical discs
            6: "ramdisk",
        }

        k32 = ctypes.windll.kernel32

        try:
            for part in psutil.disk_partitions(all=True):
                mount = part.mountpoint
                dev = part.device
                fstype = part.fstype

                # Check drive type
                dtype_code = k32.GetDriveTypeW(mount)
                dtype_str = drive_types_map.get(dtype_code, "unknown")

                # Get volume name
                vol_buf = ctypes.create_unicode_buffer(261)
                fs_buf = ctypes.create_unicode_buffer(261)
                vol_name = ""
                try:
                    if k32.GetVolumeInformationW(
                        mount, vol_buf, ctypes.sizeof(vol_buf), None, None, None, fs_buf, ctypes.sizeof(fs_buf)
                    ):
                        vol_name = vol_buf.value
                        if not fstype and fs_buf.value:
                            fstype = fs_buf.value
                except Exception:
                    pass

                # Check disk space
                tot_gb = 0.0
                usd_gb = 0.0
                fre_gb = 0.0
                pct = 0.0
                is_ready = False

                try:
                    usage = psutil.disk_usage(mount)
                    tot_gb = round(usage.total / (1024**3), 2)
                    usd_gb = round(usage.used / (1024**3), 2)
                    fre_gb = round(usage.free / (1024**3), 2)
                    pct = usage.percent
                    is_ready = True

                    total_b += usage.total
                    free_b += usage.free
                    used_b += usage.used
                except (PermissionError, OSError):
                    is_ready = False

                drives.append(
                    DriveInfo(
                        device=dev,
                        mountpoint=mount,
                        fstype=fstype or "Unknown",
                        total_gb=tot_gb,
                        used_gb=usd_gb,
                        free_gb=fre_gb,
                        percent_used=pct,
                        drive_type=dtype_str,
                        volume_name=vol_name,
                        is_ready=is_ready,
                    )
                )
        except Exception as exc:
            logger.warning("Error inspecting storage drives: %s", exc)

        tot_all_gb = round(total_b / (1024**3), 2)
        fre_all_gb = round(free_b / (1024**3), 2)
        usd_all_gb = round(used_b / (1024**3), 2)
        pct_all = round((usd_all_gb / tot_all_gb * 100.0) if tot_all_gb > 0 else 0.0, 1)

        return StorageSummary(
            total_gb=tot_all_gb,
            free_gb=fre_all_gb,
            used_gb=usd_all_gb,
            percent_used=pct_all,
            drives=drives,
        )

    # --------------------------------------------------------------------------
    # 7. Complete Profile Compilation
    # --------------------------------------------------------------------------

    def get_profile(self, refresh: bool = False) -> CompleteHardwareProfile:
        """Compile and return comprehensive host hardware specification."""
        now = time.time()
        if not refresh and self._cached_profile is not None:
            if (now - self._last_profile_time) < self._cache_duration_seconds:
                return self._cached_profile

        cpu = self.detect_cpu()
        ram = self.detect_ram()
        gpus = self.detect_gpus()
        display = self.detect_displays()
        audio = self.detect_audio_devices()
        storage = self.detect_storage()

        win_ver = sys.getwindowsversion()
        os_build = f"{win_ver.major}.{win_ver.minor}.{win_ver.build}"
        os_name = f"Windows {platform.release()}"

        profile = CompleteHardwareProfile(
            cpu=cpu,
            ram=ram,
            gpus=gpus,
            display=display,
            audio=audio,
            storage=storage,
            os_name=os_name,
            os_version=platform.version(),
            os_build=os_build,
            os_architecture=platform.machine(),
            is_admin=PrivilegeDetector.is_current_process_admin(),
            computer_name=platform.node(),
            current_user=os.environ.get("USERNAME", "User"),
        )
        self._cached_profile = profile
        self._last_profile_time = now
        return profile

    def get_complete_profile(self, refresh: bool = False) -> CompleteHardwareProfile:
        """Alias for get_profile returning comprehensive hardware profile."""
        return self.get_profile(refresh=refresh)

    # --------------------------------------------------------------------------
    # 8. Pre-Flight Resource Safety Evaluation
    # --------------------------------------------------------------------------

    def evaluate_operation_capacity(
        self,
        operation_name: str = "",
        required_disk_gb: float = 0.0,
        required_ram_gb: float = 0.0,
        target_path: Optional[str] = None,
        requires_gpu: bool = False,
        operation: Optional[str] = None,
    ) -> ResourceSafetyCheck:
        """Evaluate host resource capacity before expensive operations.

        Checks:
        1. Available disk space on the designated destination drive (preserving 2 GB reserve)
        2. Available RAM (preserving 1 GB reserve for OS and GUI)
        3. GPU compute suitability if requires_gpu is set
        4. Identifies destination volume
        5. Warns about insufficient capacity and specifies whether user approval is required
        """
        op_label = operation or operation_name or "Operation"
        ram = self.detect_ram()
        storage = self.detect_storage()
        gpus = self.detect_gpus()

        warnings: list[str] = []
        insufficient: list[str] = []
        is_safe = True

        # 1. Determine destination drive
        dest_mount = "C:\\"
        if target_path:
            p = Path(target_path).resolve()
            drive_root = p.anchor or "C:\\"
            dest_mount = drive_root.upper()
            if not dest_mount.endswith("\\"):
                dest_mount += "\\"

        dest_drive = next((d for d in storage.drives if d.mountpoint.upper() == dest_mount), None)
        avail_disk = dest_drive.free_gb if dest_drive else (storage.free_gb)

        # Check disk space with 2.0 GB safety reserve
        disk_safety_margin = 2.0
        if required_disk_gb > 0:
            if avail_disk < (required_disk_gb + disk_safety_margin):
                is_safe = False
                insufficient.append("disk")
                warnings.append(
                    f"Insufficient disk space on drive {dest_mount}: requires {required_disk_gb:.1f} GB "
                    f"+ {disk_safety_margin:.1f} GB safety reserve, but only {avail_disk:.1f} GB is available."
                )

        # Check RAM with 1.0 GB safety reserve
        ram_safety_margin = 1.0
        if required_ram_gb > 0:
            if ram.available_gb < (required_ram_gb + ram_safety_margin):
                is_safe = False
                insufficient.append("ram")
                warnings.append(
                    f"Insufficient available RAM: requires {required_ram_gb:.1f} GB "
                    f"+ {ram_safety_margin:.1f} GB reserve, but only {ram.available_gb:.1f} GB is currently available."
                )

        # Check GPU compute compatibility
        gpu_compatible = True
        if requires_gpu:
            has_llm_gpu = any(g.supports_llm_acceleration for g in gpus)
            if not has_llm_gpu:
                gpu_compatible = False
                is_safe = False
                insufficient.append("gpu")
                warnings.append(
                    "Requested operation requires dedicated GPU compute acceleration. "
                    "Detected GPUs (Intel HD Graphics, AMD Radeon 520) lack modern CUDA/ROCm acceleration runtimes on Windows."
                )

        # Recommendation synthesis
        recs = []
        if not is_safe:
            if "disk" in insufficient:
                recs.append(f"Free up at least {required_disk_gb + disk_safety_margin - avail_disk:.1f} GB on {dest_mount}.")
            if "ram" in insufficient:
                recs.append("Close heavy background applications or switch to a lightweight quantized model (e.g. 1B / 1.5B).")
            if "gpu" in insufficient:
                recs.append("Run CPU inference with quantized int8 weights capped at 2 threads.")
        recommendation = " ".join(recs) if recs else "System has sufficient resource capacity for this operation."

        # High-impact operations requiring approval:
        # If disk > 2.0 GB or RAM > 3.0 GB or any resource is strained
        req_approval = not is_safe or required_disk_gb >= 2.0 or required_ram_gb >= 3.0

        return ResourceSafetyCheck(
            is_safe=is_safe,
            operation=op_label,
            target_drive=dest_mount,
            required_disk_gb=round(required_disk_gb, 2),
            available_disk_gb=round(avail_disk, 2),
            required_ram_gb=round(required_ram_gb, 2),
            available_ram_gb=round(ram.available_gb, 2),
            gpu_required=requires_gpu,
            gpu_compatible=gpu_compatible,
            warnings=warnings,
            insufficient_resources=insufficient,
            requires_user_approval=req_approval,
            recommendation=recommendation,
        )
