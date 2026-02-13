"""
ViewerDBX GPU Monitor Widget
============================
Real-time GPU monitoring with VBUS cache visualization.

Displays:
- GPU VRAM usage (real via pynvml or nvidia-smi)
- GPU temperature and utilization
- VBUS L1/L2/L3 cache status
- Active ML tools and agents
- Memory pressure warnings
- Emergency VRAM kill button

Integrates with:
- GPU Engine/VRAM_EMERGENCY_KILLER.py
- GPU Engine/gpu_bridge.py
- .claude/gpu_monitor_integration.py

Launch: pythonw ViewerDBX_GPU_Monitor.pyw
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import threading
import time

# Try to import pynvml for real GPU stats
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Try subprocess for fallback nvidia-smi
import subprocess

# Paths
SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / '.gpu_monitor_state.json'
VBUS_FILE = SCRIPT_DIR / '.vbus_state.json'
GPU_ENGINE_DIR = SCRIPT_DIR.parent / 'GPU Engine'
PARALLEL_ENGINE_DIR = SCRIPT_DIR.parent / 'Parallel Engine'
VRAM_KILLER = GPU_ENGINE_DIR / 'VRAM_EMERGENCY_KILLER.py'
VBUS_CACHE_DIR = SCRIPT_DIR.parent / '.vbus_cache'
VBUS_STATE_FILE = SCRIPT_DIR / '.vbus_state.json'  # From VBUS Resource Protector
ML_TOOLS_STATE_FILE = SCRIPT_DIR / '.ml_tools_state.json'  # ML tool usage tracking

# Add Parallel Engine to path for system detection
sys.path.insert(0, str(PARALLEL_ENGINE_DIR))

# RTX 3080 Ti specs from CLAUDE.md
GPU_VRAM_TOTAL_MB = 12288  # 12GB
GPU_VRAM_TOTAL_GB = 12.0
GPU_NAME = "RTX 3080 Ti"

# Ryzen 9 3950X specs from Parallel Engine CLAUDE.md
CPU_NAME = "Ryzen 9 3950X"
CPU_CORES = 16
CPU_THREADS = 32
CPU_L3_CACHE_MB = 64  # 4x CCX x 16MB

# System RAM
RAM_TOTAL_GB = 128

# System SSD
SSD_TOTAL_TB = 4.0

# Thresholds (matching VRAM_EMERGENCY_KILLER.py)
VRAM_CRITICAL = 500    # MB - trigger emergency clear
VRAM_WARNING = 1000    # MB - start aggressive eviction
VRAM_SAFE = 2000       # MB - normal operation

# RAM thresholds
RAM_WARNING_GB = 16    # Less than 16GB free = warning
RAM_CRITICAL_GB = 8    # Less than 8GB free = critical

# VBUS Cache tiers from CLAUDE.md
VBUS_L1_ENTRIES = 64    # GPU VRAM (Hot)
VBUS_L2_ENTRIES = 256   # Pinned RAM (Staging)
VBUS_L3_ENTRIES = 1024  # Disk (Cold)

# Colors
COLORS = {
    'bg': '#0d1117',
    'panel': '#161b22',
    'border': '#30363d',
    'text': '#c9d1d9',
    'text_dim': '#8b949e',
    'green': '#3fb950',
    'yellow': '#d29922',
    'orange': '#db6d28',
    'red': '#f85149',
    'blue': '#58a6ff',
    'purple': '#a371f7',
    'cyan': '#39c5cf',
}

# Agent colors
AGENT_COLORS = {
    'Fiona': '#93C5FD',
    'ProQ': '#6EE7B7',
    'JamesHunt': '#67E8F9',
    'Sally': '#6EE7B7',
    'Randy': '#6EE7B7',
    'Madge': '#A78BFA',
    'MadgeMatch': '#A78BFA',
    'RandyReplit': '#6EE7B7',
}

# ML Tools - display names kept short for clean columns
ML_TOOLS = [
    'cuDF',
    'XGBoost',
    'cuML',
    'Dask-RAPIDS',
    'cuGraph',
    'CatBoost',
    'PyTorch',
    'RandomForest',
    'cuFFT',
    'cuSOLVER',
    'cuBLAS',
]


class SystemStats:
    """Get CPU and RAM statistics including cores/threads and L3 cache."""

    @staticmethod
    def get_stats():
        """Return dict with CPU and RAM stats."""
        stats = {
            'cpu_percent': 0,
            'cores_active': 0,
            'threads_active': 0,
            'l3_cache_used_mb': 0,
            'ram_total_gb': RAM_TOTAL_GB,
            'ram_used_gb': 0,
            'ram_free_gb': RAM_TOTAL_GB,
            'ram_percent': 0,
            'real_data': False,
        }

        # Try Windows-specific memory detection
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

                stats['ram_total_gb'] = stat.ullTotalPhys / (1024**3)
                stats['ram_free_gb'] = stat.ullAvailPhys / (1024**3)
                stats['ram_used_gb'] = stats['ram_total_gb'] - stats['ram_free_gb']
                stats['ram_percent'] = stat.dwMemoryLoad
                stats['real_data'] = True
            except Exception:
                pass

        # Get CPU usage via wmic (lightweight)
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'loadpercentage', '/format:csv'],
                capture_output=True, text=True, timeout=2,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if len(lines) > 1:
                parts = lines[-1].split(',')
                if len(parts) >= 2 and parts[1].isdigit():
                    stats['cpu_percent'] = int(parts[1])
                    stats['real_data'] = True
        except Exception:
            pass

        # Estimate active cores/threads based on CPU load
        # This is an approximation - actual thread count would require per-core sampling
        cpu_pct = stats['cpu_percent']
        if cpu_pct > 0:
            # Estimate threads in use based on overall load
            stats['threads_active'] = max(1, int((cpu_pct / 100.0) * CPU_THREADS))
            # Cores active = at least threads_active / 2 (due to hyperthreading)
            stats['cores_active'] = max(1, min(CPU_CORES, (stats['threads_active'] + 1) // 2))

        # Estimate L3 cache usage based on memory pressure and CPU activity
        # L3 cache is shared, so estimate based on overall system activity
        if stats['real_data']:
            # Higher RAM usage = likely higher L3 cache pressure
            ram_pressure = min(1.0, stats['ram_used_gb'] / stats['ram_total_gb'])
            cpu_pressure = cpu_pct / 100.0
            # Weighted estimate: 60% RAM pressure + 40% CPU activity
            cache_usage_pct = (ram_pressure * 0.6 + cpu_pressure * 0.4)
            stats['l3_cache_used_mb'] = int(cache_usage_pct * CPU_L3_CACHE_MB)

        return stats


class GPUStats:
    """Get real GPU statistics using pynvml or nvidia-smi fallback."""

    def __init__(self):
        self.initialized = False
        self.handle = None

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.initialized = True
            except Exception as e:
                print(f"pynvml init failed: {e}")

    def get_stats(self):
        """Return dict with GPU stats."""
        stats = {
            'vram_used_mb': 0,
            'vram_total_mb': GPU_VRAM_TOTAL_MB,
            'vram_percent': 0,
            'temperature': 0,
            'utilization': 0,
            'power_draw': 0,
            'power_limit': 350,
            'name': GPU_NAME,
            'real_data': False,
        }

        # Try pynvml first
        if self.initialized and self.handle:
            try:
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                stats['vram_used_mb'] = mem_info.used // (1024 * 1024)
                stats['vram_total_mb'] = mem_info.total // (1024 * 1024)
                stats['vram_percent'] = (mem_info.used / mem_info.total) * 100

                # Temperature
                stats['temperature'] = pynvml.nvmlDeviceGetTemperature(
                    self.handle, pynvml.NVML_TEMPERATURE_GPU
                )

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                stats['utilization'] = util.gpu

                # Power
                try:
                    stats['power_draw'] = pynvml.nvmlDeviceGetPowerUsage(self.handle) // 1000
                    stats['power_limit'] = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) // 1000
                except:
                    pass

                # Name
                try:
                    stats['name'] = pynvml.nvmlDeviceGetName(self.handle)
                    if isinstance(stats['name'], bytes):
                        stats['name'] = stats['name'].decode('utf-8')
                except:
                    pass

                stats['real_data'] = True
                return stats

            except Exception as e:
                print(f"pynvml read error: {e}")

        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit,name',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 6:
                    stats['vram_used_mb'] = int(float(parts[0]))
                    stats['vram_total_mb'] = int(float(parts[1]))
                    stats['vram_percent'] = (stats['vram_used_mb'] / stats['vram_total_mb']) * 100
                    stats['temperature'] = int(float(parts[2]))
                    stats['utilization'] = int(float(parts[3]))
                    stats['power_draw'] = int(float(parts[4]))
                    stats['power_limit'] = int(float(parts[5]))
                    if len(parts) >= 7:
                        stats['name'] = parts[6]
                    stats['real_data'] = True
        except Exception as e:
            print(f"nvidia-smi fallback failed: {e}")

        return stats

    def shutdown(self):
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class ViewerDBXMonitor(tk.Tk):
    """Main GPU Monitor Widget."""

    def __init__(self):
        super().__init__()

        self.title("AI ML VBUS Dashboard")
        self.configure(bg='#00aaff')  # Blue glow outer color
        self.attributes('-topmost', True)
        self.overrideredirect(True)  # Remove window frame for custom border

        # Blue glowing border - outer glow frame
        self.glow_border = 4  # Glow thickness

        # Window size and position (center of screen) - starts fully expanded
        width, height = 420, 880
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Draggable window (since we removed the title bar)
        self._drag_data = {'x': 0, 'y': 0}
        self.bind('<Button-1>', self._start_drag)
        self.bind('<B1-Motion>', self._do_drag)

        # GPU stats reader
        self.gpu = GPUStats()

        # State
        self.running = True
        self.tool_states = {}
        self.vbus_state = {'active': False, 'hits': 0, 'l1': 0, 'l2': 0, 'l3': 0}
        self.session_start_time = datetime.now()  # Track session start for timer
        self.gpu_is_active = False  # Track if GPU is actually being used

        # Build UI
        self._build_ui()

        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Right-click context menu
        self._create_context_menu()
        self.bind("<Button-3>", self._show_context_menu)

    def _create_context_menu(self):
        """Create right-click context menu with helpful options."""
        self.context_menu = tk.Menu(self, tearoff=0, bg=COLORS['panel'], fg=COLORS['text'],
                                     activebackground=COLORS['blue'], activeforeground='white',
                                     font=('Consolas', 10))

        self.context_menu.add_command(label="📖 README / Help", command=self._open_readme)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="🔄 Refresh Now", command=self._force_refresh)
        self.context_menu.add_command(label="🧹 Clear VBUS Cache", command=self._clear_vbus_cache)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="📊 Reset Session Timer", command=self._reset_session)
        self.context_menu.add_command(label="📋 Copy Stats to Clipboard", command=self._copy_stats)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="⚡ Emergency GPU Kill", command=self._emergency_kill)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="📌 Toggle Always On Top", command=self._toggle_topmost)
        self.context_menu.add_command(label="🔽 Collapse All Panels", command=self._collapse_panels)
        self.context_menu.add_command(label="🔼 Expand All Panels", command=self._expand_panels)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="❌ Close Dashboard", command=self._on_close)

    def _show_context_menu(self, event):
        """Show the context menu at mouse position."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _open_readme(self):
        """Open the README file in default text editor."""
        readme_path = SCRIPT_DIR / 'WIDGET_README.md'
        try:
            if os.name == 'nt':
                os.startfile(readme_path)
            else:
                subprocess.run(['xdg-open', str(readme_path)])
            self.footer.config(text="README opened", fg=COLORS['green'])
        except Exception as e:
            self.footer.config(text=f"Cannot open README: {e}", fg=COLORS['red'])

    def _force_refresh(self):
        """Force immediate refresh of all data."""
        self._refresh_display()
        self.footer.config(text="Refreshed!", fg=COLORS['green'])

    def _clear_vbus_cache(self):
        """Clear VBUS cache and update state file."""
        try:
            # Reset VBUS state
            vbus_state = {
                'vbus_running': False, 'protection_level': 'normal', 'last_updated': '',
                'l1_allocated_mb': 0.0, 'l1_entries': 0, 'l2_allocated_gb': 0.0, 'l2_entries': 0,
                'l3_allocated_gb': 0.0, 'l3_entries': 0, 'vram_free_mb': GPU_VRAM_TOTAL_MB,
                'vram_total_mb': GPU_VRAM_TOTAL_MB, 'ram_free_gb': RAM_TOTAL_GB,
                'ram_total_gb': RAM_TOTAL_GB, 'total_hits': 0, 'l1_hits': 0, 'l2_hits': 0,
                'l3_hits': 0, 'evictions_today': 0, 'emergency_clears_today': 0
            }
            with open(VBUS_STATE_FILE, 'w') as f:
                json.dump(vbus_state, f, indent=2)
            self.footer.config(text="VBUS cache cleared!", fg=COLORS['green'])
        except Exception as e:
            self.footer.config(text=f"Clear failed: {e}", fg=COLORS['red'])

    def _reset_session(self):
        """Reset the session timer."""
        self.session_start_time = datetime.now()
        self.footer.config(text="Session timer reset!", fg=COLORS['green'])

    def _copy_stats(self):
        """Copy current stats to clipboard."""
        try:
            gpu_stats = self.gpu.get_stats()
            stats_text = f"""AI ML VBUS Dashboard Stats
========================
GPU: {GPU_NAME}
VRAM: {gpu_stats.get('memory_used', 0):.0f} / {gpu_stats.get('memory_total', GPU_VRAM_TOTAL_MB):.0f} MB
Temp: {gpu_stats.get('temperature', 0)}°C
Util: {gpu_stats.get('utilization', 0)}%
Session: {datetime.now() - self.session_start_time}
"""
            self.clipboard_clear()
            self.clipboard_append(stats_text)
            self.footer.config(text="Stats copied to clipboard!", fg=COLORS['green'])
        except Exception as e:
            self.footer.config(text=f"Copy failed: {e}", fg=COLORS['red'])

    def _toggle_topmost(self):
        """Toggle always-on-top mode."""
        current = self.attributes('-topmost')
        self.attributes('-topmost', not current)
        status = "ON" if not current else "OFF"
        self.footer.config(text=f"Always on top: {status}", fg=COLORS['cyan'])

    def _collapse_panels(self):
        """Collapse all panels."""
        if self.panels_expanded:
            self._toggle_panels()

    def _expand_panels(self):
        """Expand all panels."""
        if not self.panels_expanded:
            self._toggle_panels()

    def _start_drag(self, event):
        """Start window drag."""
        self._drag_data['x'] = event.x
        self._drag_data['y'] = event.y

    def _do_drag(self, event):
        """Handle window dragging."""
        x = self.winfo_x() + (event.x - self._drag_data['x'])
        y = self.winfo_y() + (event.y - self._drag_data['y'])
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Build the UI components with collapsible panels and blue glow border."""
        # Blue glow outer border (creates the glow effect)
        glow_frame = tk.Frame(self, bg='#00aaff', padx=3, pady=3)
        glow_frame.pack(fill='both', expand=True)

        # Inner border (slightly darker blue)
        inner_glow = tk.Frame(glow_frame, bg='#0088dd', padx=1, pady=1)
        inner_glow.pack(fill='both', expand=True)

        # Main container with dark background
        self.main_frame = tk.Frame(inner_glow, bg=COLORS['bg'], padx=10, pady=10)
        self.main_frame.pack(fill='both', expand=True)

        # Header
        header = tk.Frame(self.main_frame, bg=COLORS['bg'])
        header.pack(fill='x', pady=(0, 10))

        # Title and copyright container
        title_frame = tk.Frame(header, bg=COLORS['bg'])
        title_frame.pack(side='left')

        tk.Label(title_frame, text="AI ML VBUS Dashboard", font=('Consolas', 11),
                 fg=COLORS['blue'], bg=COLORS['bg']).pack(anchor='w')
        tk.Label(title_frame, text="© Framecore Inc", font=('Consolas', 11),
                 fg=COLORS['text_dim'], bg=COLORS['bg']).pack(anchor='w')

        # Close button (X) - since we removed window frame
        close_btn = tk.Button(header, text="X", font=('Consolas', 9, 'bold'),
                              fg='white', bg=COLORS['border'], activebackground=COLORS['red'],
                              relief='flat', padx=6, pady=2,
                              command=self._on_close)
        close_btn.pack(side='right', padx=(5, 0))

        # Emergency kill button
        self.kill_btn = tk.Button(header, text="KILL", font=('Consolas', 9, 'bold'),
                                   fg='white', bg=COLORS['red'], activebackground='#c41e3a',
                                   relief='flat', padx=8, pady=2,
                                   command=self._emergency_kill)
        self.kill_btn.pack(side='right', padx=(5, 0))

        self.status_dot = tk.Label(header, text="●", font=('Consolas', 11),
                                    fg=COLORS['green'], bg=COLORS['bg'])
        self.status_dot.pack(side='right')

        # GPU Panel (always visible)
        self._build_gpu_panel()

        # VRAM Gauge (always visible)
        self._build_vram_gauge()

        # Collapsible panels container
        self.collapsible_frame = tk.Frame(self.main_frame, bg=COLORS['bg'])
        self.collapsible_frame.pack(fill='x')

        # Track collapsed state (start expanded so users see everything)
        self.panels_expanded = True

        # Expand/Collapse button
        self.expand_btn = tk.Button(self.collapsible_frame, text="▲ Less Info",
                                     font=('Consolas', 11), fg=COLORS['text_dim'],
                                     bg=COLORS['panel'], activebackground=COLORS['border'],
                                     relief='flat', cursor='hand2',
                                     command=self._toggle_panels)
        self.expand_btn.pack(fill='x', pady=(5, 0))

        # Collapsible panels container (starts visible)
        self.hidden_panels = tk.Frame(self.collapsible_frame, bg=COLORS['bg'])
        self.hidden_panels.pack(fill='x', pady=(5, 0))  # Start visible

        # Build hidden panels inside container
        self._build_system_panel_collapsible()
        self._build_vbus_panel_collapsible()
        self._build_tools_panel_collapsible()

        # Footer
        self.footer = tk.Label(self.main_frame, text="Waiting for data...",
                               font=('Consolas', 11), fg=COLORS['text_dim'], bg=COLORS['bg'])
        self.footer.pack(side='bottom', pady=(5, 0))

    def _toggle_panels(self):
        """Toggle visibility of collapsible panels."""
        if self.panels_expanded:
            # Collapse
            self.hidden_panels.pack_forget()
            self.expand_btn.config(text="▼ More Info")
            self.panels_expanded = False
            # Resize window smaller
            self.geometry(f"420x320+{self.winfo_x()}+{self.winfo_y()}")
        else:
            # Expand
            self.hidden_panels.pack(fill='x', pady=(5, 0))
            self.expand_btn.config(text="▲ Less Info")
            self.panels_expanded = True
            # Resize window larger
            self.geometry(f"420x880+{self.winfo_x()}+{self.winfo_y()}")

    def _build_gpu_panel(self):
        """GPU stats panel."""
        panel = tk.LabelFrame(self.main_frame, text=" GPU ", font=('Consolas', 11),
                              fg=COLORS['cyan'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='x', pady=(0, 8))

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        # GPU Name and stats on single row
        stats_frame = tk.Frame(inner, bg=COLORS['panel'])
        stats_frame.pack(fill='x')

        # GPU Name on left
        self.gpu_name_label = tk.Label(stats_frame, text=GPU_NAME, font=('Consolas', 11),
                                        fg=COLORS['text'], bg=COLORS['panel'])
        self.gpu_name_label.pack(side='left')

        # Stats on right (inline labels with values)
        right_stats = tk.Frame(stats_frame, bg=COLORS['panel'])
        right_stats.pack(side='right')

        # Temperature: TEMP 45°C
        tk.Label(right_stats, text="TEMP", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='left', padx=(0, 2))
        self.temp_label = tk.Label(right_stats, text="--°C", font=('Consolas', 11),
                                    fg=COLORS['green'], bg=COLORS['panel'])
        self.temp_label.pack(side='left', padx=(0, 10))

        # Utilization: UTIL 23%
        tk.Label(right_stats, text="UTIL", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='left', padx=(0, 2))
        self.util_label = tk.Label(right_stats, text="--%", font=('Consolas', 11),
                                    fg=COLORS['green'], bg=COLORS['panel'])
        self.util_label.pack(side='left', padx=(0, 10))

        # Power: POWER 150W
        tk.Label(right_stats, text="POWER", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='left', padx=(0, 2))
        self.power_label = tk.Label(right_stats, text="--W", font=('Consolas', 11),
                                     fg=COLORS['green'], bg=COLORS['panel'])
        self.power_label.pack(side='left')

    def _build_vram_gauge(self):
        """VRAM usage gauge."""
        panel = tk.LabelFrame(self.main_frame, text=" VRAM ", font=('Consolas', 11),
                              fg=COLORS['purple'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='x', pady=(0, 8))

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        # Usage text
        usage_frame = tk.Frame(inner, bg=COLORS['panel'])
        usage_frame.pack(fill='x')

        self.vram_label = tk.Label(usage_frame, text="0 / 12288 MB",
                                    font=('Consolas', 11),
                                    fg=COLORS['text'], bg=COLORS['panel'])
        self.vram_label.pack(side='left')

        self.vram_percent_label = tk.Label(usage_frame, text="0%",
                                            font=('Consolas', 11),
                                            fg=COLORS['green'], bg=COLORS['panel'])
        self.vram_percent_label.pack(side='right')

        # Progress bar canvas
        self.vram_canvas = tk.Canvas(inner, height=24, bg=COLORS['bg'],
                                      highlightthickness=1, highlightbackground=COLORS['border'])
        self.vram_canvas.pack(fill='x', pady=(8, 0))

        # Threshold markers
        thresh_frame = tk.Frame(inner, bg=COLORS['panel'])
        thresh_frame.pack(fill='x')
        tk.Label(thresh_frame, text="0%", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='left')
        tk.Label(thresh_frame, text="70%", font=('Consolas', 11), fg=COLORS['yellow'],
                 bg=COLORS['panel']).pack(side='left', padx=(95, 0))
        tk.Label(thresh_frame, text="90%", font=('Consolas', 11), fg=COLORS['red'],
                 bg=COLORS['panel']).pack(side='left', padx=(50, 0))
        tk.Label(thresh_frame, text="100%", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='right')

    def _build_system_panel_collapsible(self):
        """CPU and RAM stats panel (collapsible version)."""
        panel = tk.LabelFrame(self.hidden_panels, text=" SYSTEM ", font=('Consolas', 11),
                              fg=COLORS['yellow'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='x', pady=(5, 5))

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        # Single row: CPU name on left, stats on right (inline)
        stats_frame = tk.Frame(inner, bg=COLORS['panel'])
        stats_frame.pack(fill='x')

        # CPU Name on left
        self.cpu_name_label = tk.Label(stats_frame, text=CPU_NAME,
                                        font=('Consolas', 11),
                                        fg=COLORS['text'], bg=COLORS['panel'])
        self.cpu_name_label.pack(side='left')

        # Stats on right (inline labels with values)
        right_stats = tk.Frame(stats_frame, bg=COLORS['panel'])
        right_stats.pack(side='right')

        # LOAD 23%
        tk.Label(right_stats, text="LOAD", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='left', padx=(0, 2))
        self.cpu_label = tk.Label(right_stats, text="--%", font=('Consolas', 11),
                                   fg=COLORS['green'], bg=COLORS['panel'])
        self.cpu_label.pack(side='left', padx=(0, 10))

        # CORES 8/16/32
        tk.Label(right_stats, text="CORES", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='left', padx=(0, 2))
        self.cores_label = tk.Label(right_stats, text=f"0/{CPU_CORES}/{CPU_THREADS}",
                                     font=('Consolas', 11),
                                     fg=COLORS['cyan'], bg=COLORS['panel'])
        self.cores_label.pack(side='left', padx=(0, 10))

        # L3 0/64MB
        tk.Label(right_stats, text="L3", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='left', padx=(0, 2))
        self.l3_cache_label = tk.Label(right_stats, text=f"0/{CPU_L3_CACHE_MB}MB",
                                        font=('Consolas', 11),
                                        fg=COLORS['purple'], bg=COLORS['panel'])
        self.l3_cache_label.pack(side='left')

        # RAM labels kept as placeholders for compatibility
        self.ram_used_label = None
        self.ram_free_label = None

    def _build_system_panel(self):
        """CPU and RAM stats panel with cores/threads and L3 cache (legacy - not used)."""
        panel = tk.LabelFrame(self.main_frame, text=" SYSTEM ", font=('Consolas', 10, 'bold'),
                              fg=COLORS['yellow'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='x', pady=(0, 10))

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        # Top row: CPU name
        top_frame = tk.Frame(inner, bg=COLORS['panel'])
        top_frame.pack(fill='x')

        self.cpu_name_label = tk.Label(top_frame, text=CPU_NAME,
                                        font=('Consolas', 9, 'bold'),
                                        fg=COLORS['text'], bg=COLORS['panel'])
        self.cpu_name_label.pack(side='left')

        # CPU stats row (cores/threads, L3 cache)
        cpu_stats_frame = tk.Frame(inner, bg=COLORS['panel'])
        cpu_stats_frame.pack(fill='x', pady=(4, 0))

        # CPU Usage %
        cpu_pct_frame = tk.Frame(cpu_stats_frame, bg=COLORS['panel'])
        cpu_pct_frame.pack(side='left', expand=True)
        tk.Label(cpu_pct_frame, text="LOAD", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack()
        self.cpu_label = tk.Label(cpu_pct_frame, text="--%", font=('Consolas', 11, 'bold'),
                                   fg=COLORS['green'], bg=COLORS['panel'])
        self.cpu_label.pack()

        # Cores/Threads in use
        cores_frame = tk.Frame(cpu_stats_frame, bg=COLORS['panel'])
        cores_frame.pack(side='left', expand=True)
        tk.Label(cores_frame, text="CORES/THR", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack()
        self.cores_label = tk.Label(cores_frame, text=f"0/{CPU_CORES}/{CPU_THREADS}",
                                     font=('Consolas', 10, 'bold'),
                                     fg=COLORS['cyan'], bg=COLORS['panel'])
        self.cores_label.pack()

        # L3 Cache usage
        l3_frame = tk.Frame(cpu_stats_frame, bg=COLORS['panel'])
        l3_frame.pack(side='left', expand=True)
        tk.Label(l3_frame, text="L3 CACHE", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack()
        self.l3_cache_label = tk.Label(l3_frame, text=f"0/{CPU_L3_CACHE_MB}MB",
                                        font=('Consolas', 10, 'bold'),
                                        fg=COLORS['purple'], bg=COLORS['panel'])
        self.l3_cache_label.pack()

        # RAM stats row
        ram_frame = tk.Frame(inner, bg=COLORS['panel'])
        ram_frame.pack(fill='x', pady=(6, 0))

        # RAM Used
        ram_used_frame = tk.Frame(ram_frame, bg=COLORS['panel'])
        ram_used_frame.pack(side='left', expand=True)
        tk.Label(ram_used_frame, text="RAM USED", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack()
        self.ram_used_label = tk.Label(ram_used_frame, text="--GB", font=('Consolas', 12, 'bold'),
                                        fg=COLORS['green'], bg=COLORS['panel'])
        self.ram_used_label.pack()

        # RAM Free
        ram_free_frame = tk.Frame(ram_frame, bg=COLORS['panel'])
        ram_free_frame.pack(side='left', expand=True)
        tk.Label(ram_free_frame, text="RAM FREE", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack()
        self.ram_free_label = tk.Label(ram_free_frame, text="--GB", font=('Consolas', 12, 'bold'),
                                        fg=COLORS['green'], bg=COLORS['panel'])
        self.ram_free_label.pack()

    def _build_vbus_panel_collapsible(self):
        """VBUS cache tier panel (collapsible version)."""
        # Panel with custom header frame for ACTIVE/INACTIVE indicator
        panel = tk.LabelFrame(self.hidden_panels, text="", font=('Consolas', 11),
                              fg=COLORS['orange'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='x', pady=(0, 5))

        # Custom header with title and ACTIVE/INACTIVE
        header_frame = tk.Frame(panel, bg=COLORS['panel'])
        header_frame.pack(fill='x', padx=10, pady=(6, 0))

        tk.Label(header_frame, text="VBUS PROTECTED CACHE", font=('Consolas', 11),
                 fg=COLORS['orange'], bg=COLORS['panel']).pack(side='left')

        # ACTIVE/INACTIVE indicator (bright green or light gray)
        self.vbus_active_label = tk.Label(header_frame, text="INACTIVE",
                                           font=('Consolas', 11),
                                           fg=COLORS['text_dim'], bg=COLORS['panel'])
        self.vbus_active_label.pack(side='left', padx=(10, 0))

        self.vbus_hits = tk.Label(header_frame, text="0 hits",
                                   font=('Consolas', 11),
                                   fg=COLORS['text_dim'], bg=COLORS['panel'])
        self.vbus_hits.pack(side='right')

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        # Column headers for cache values (35% smaller font = ~7pt)
        header_frame = tk.Frame(inner, bg=COLORS['panel'])
        header_frame.pack(fill='x', pady=(6, 2))
        tk.Label(header_frame, text="in use", font=('Consolas', 7), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='right', padx=(0, 2))
        tk.Label(header_frame, text="alloc", font=('Consolas', 7), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='right', padx=(0, 12))
        tk.Label(header_frame, text="max avail", font=('Consolas', 7), fg=COLORS['text_dim'],
                 bg=COLORS['panel']).pack(side='right', padx=(0, 8))

        # Cache tiers: in-use / allocated / max-available
        tiers_frame = tk.Frame(inner, bg=COLORS['panel'])
        tiers_frame.pack(fill='x', pady=(0, 0))

        # L1 Cache (VRAM) - in-use / allocated / max (GB)
        l1_frame = tk.Frame(tiers_frame, bg=COLORS['panel'])
        l1_frame.pack(fill='x', pady=2)
        tk.Label(l1_frame, text="L1 Cache (VRAM)", font=('Consolas', 11), fg=COLORS['cyan'],
                 bg=COLORS['panel'], anchor='w').pack(side='left')
        self.l1_bar = tk.Canvas(l1_frame, height=12, bg=COLORS['bg'], width=60,
                                 highlightthickness=1, highlightbackground=COLORS['border'])
        self.l1_bar.pack(side='left', padx=(5, 5))
        self.l1_label = tk.Label(l1_frame, text="0.0 / 0.0 / 12", font=('Consolas', 11),
                                  fg=COLORS['text_dim'], bg=COLORS['panel'], anchor='e')
        self.l1_label.pack(side='right')

        # L2 Cache (SysRAM) - active / allocated / max (GB)
        l2_frame = tk.Frame(tiers_frame, bg=COLORS['panel'])
        l2_frame.pack(fill='x', pady=2)
        tk.Label(l2_frame, text="L2 Cache (SysRAM)", font=('Consolas', 11), fg=COLORS['yellow'],
                 bg=COLORS['panel'], anchor='w').pack(side='left')
        self.l2_bar = tk.Canvas(l2_frame, height=12, bg=COLORS['bg'], width=60,
                                 highlightthickness=1, highlightbackground=COLORS['border'])
        self.l2_bar.pack(side='left', padx=(5, 5))
        self.l2_label = tk.Label(l2_frame, text="0.0 / 0.0 / 128", font=('Consolas', 11),
                                  fg=COLORS['text_dim'], bg=COLORS['panel'], anchor='e')
        self.l2_label.pack(side='right')

        # L3 Cache (SSD) - active / allocated / max (TB)
        l3_frame = tk.Frame(tiers_frame, bg=COLORS['panel'])
        l3_frame.pack(fill='x', pady=2)
        tk.Label(l3_frame, text="L3 Cache (SSD)", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel'], anchor='w').pack(side='left')
        self.l3_bar = tk.Canvas(l3_frame, height=12, bg=COLORS['bg'], width=60,
                                 highlightthickness=1, highlightbackground=COLORS['border'])
        self.l3_bar.pack(side='left', padx=(5, 5))
        self.l3_label = tk.Label(l3_frame, text="0.0 / 0.0 / 4", font=('Consolas', 11),
                                  fg=COLORS['text_dim'], bg=COLORS['panel'], anchor='e')
        self.l3_label.pack(side='right')

    def _build_vbus_panel(self):
        """VBUS cache tier panel with protected allocation limits (legacy - not used)."""
        panel = tk.LabelFrame(self.main_frame, text=" VBUS PROTECTED CACHE ", font=('Consolas', 10, 'bold'),
                              fg=COLORS['orange'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='x', pady=(0, 10))

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=8)
        inner.pack(fill='x')

        # VBUS status row
        status_frame = tk.Frame(inner, bg=COLORS['panel'])
        status_frame.pack(fill='x')

        self.vbus_status = tk.Label(status_frame, text="● INACTIVE",
                                     font=('Consolas', 10, 'bold'),
                                     fg=COLORS['text_dim'], bg=COLORS['panel'])
        self.vbus_status.pack(side='left')

        # Protection level indicator
        self.protection_level = tk.Label(status_frame, text="NORMAL",
                                          font=('Consolas', 8, 'bold'),
                                          fg=COLORS['green'], bg=COLORS['panel'])
        self.protection_level.pack(side='left', padx=(10, 0))

        self.vbus_hits = tk.Label(status_frame, text="0 hits",
                                   font=('Consolas', 11),
                                   fg=COLORS['text_dim'], bg=COLORS['panel'])
        self.vbus_hits.pack(side='right')

        # Cache tiers with memory allocation display
        tiers_frame = tk.Frame(inner, bg=COLORS['panel'])
        tiers_frame.pack(fill='x', pady=(8, 0))

        # L1 (VRAM) - shows GB allocated / 12GB total VRAM
        l1_frame = tk.Frame(tiers_frame, bg=COLORS['panel'])
        l1_frame.pack(fill='x', pady=2)
        tk.Label(l1_frame, text="L1 VRAM", font=('Consolas', 11), fg=COLORS['cyan'],
                 bg=COLORS['panel'], width=8, anchor='w').pack(side='left')
        self.l1_bar = tk.Canvas(l1_frame, height=12, bg=COLORS['bg'], width=140,
                                 highlightthickness=1, highlightbackground=COLORS['border'])
        self.l1_bar.pack(side='left', padx=(5, 5))
        self.l1_label = tk.Label(l1_frame, text="0 / 12 GB", font=('Consolas', 11),
                                  fg=COLORS['text_dim'], bg=COLORS['panel'], width=12, anchor='e')
        self.l1_label.pack(side='right')

        # L2 (RAM) - shows GB allocated / 128GB total RAM
        l2_frame = tk.Frame(tiers_frame, bg=COLORS['panel'])
        l2_frame.pack(fill='x', pady=2)
        tk.Label(l2_frame, text="L2 RAM", font=('Consolas', 11), fg=COLORS['yellow'],
                 bg=COLORS['panel'], width=8, anchor='w').pack(side='left')
        self.l2_bar = tk.Canvas(l2_frame, height=12, bg=COLORS['bg'], width=140,
                                 highlightthickness=1, highlightbackground=COLORS['border'])
        self.l2_bar.pack(side='left', padx=(5, 5))
        self.l2_label = tk.Label(l2_frame, text="0 / 128 GB", font=('Consolas', 11),
                                  fg=COLORS['text_dim'], bg=COLORS['panel'], width=12, anchor='e')
        self.l2_label.pack(side='right')

        # L3 (SSD) - shows GB allocated / 4TB total SSD
        l3_frame = tk.Frame(tiers_frame, bg=COLORS['panel'])
        l3_frame.pack(fill='x', pady=2)
        tk.Label(l3_frame, text="L3 SSD", font=('Consolas', 11), fg=COLORS['text_dim'],
                 bg=COLORS['panel'], width=8, anchor='w').pack(side='left')
        self.l3_bar = tk.Canvas(l3_frame, height=12, bg=COLORS['bg'], width=140,
                                 highlightthickness=1, highlightbackground=COLORS['border'])
        self.l3_bar.pack(side='left', padx=(5, 5))
        self.l3_label = tk.Label(l3_frame, text="0 / 4 TB", font=('Consolas', 11),
                                  fg=COLORS['text_dim'], bg=COLORS['panel'], width=12, anchor='e')
        self.l3_label.pack(side='right')

    def _build_tools_panel_collapsible(self):
        """ML Tools status panel (collapsible version) with single column layout."""
        # Panel with title in border like other panels
        panel = tk.LabelFrame(self.hidden_panels, text=" ML TOOLS ", font=('Consolas', 11),
                              fg=COLORS['green'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='x', pady=(0, 5))

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        # Status row with ACTIVE/INACTIVE
        status_row = tk.Frame(inner, bg=COLORS['panel'])
        status_row.pack(fill='x', pady=(0, 5))

        self.ml_active_label = tk.Label(status_row, text="INACTIVE",
                                         font=('Consolas', 11),
                                         fg=COLORS['text_dim'], bg=COLORS['panel'])
        self.ml_active_label.pack(side='left')

        self.ml_total_label = tk.Label(status_row, text="0 in use",
                                        font=('Consolas', 11),
                                        fg=COLORS['text_dim'], bg=COLORS['panel'])
        self.ml_total_label.pack(side='right')

        # Single column layout for tools - each row: Tool name | Active | Agent
        tools_container = tk.Frame(inner, bg=COLORS['panel'])
        tools_container.pack(fill='x')

        self.tool_labels = {}
        self.tool_stats = {}
        self.tool_agents = {}

        for tool in ML_TOOLS:
            tool_row = tk.Frame(tools_container, bg=COLORS['panel'])
            tool_row.pack(fill='x', pady=1)

            # Tool name on left
            name_label = tk.Label(tool_row, text=tool, font=('Consolas', 11),
                                   fg=COLORS['text'], bg=COLORS['panel'],
                                   width=12, anchor='w')
            name_label.pack(side='left')

            # Active/idle status in middle
            stats_label = tk.Label(tool_row, text="idle", font=('Consolas', 11),
                                    fg=COLORS['text_dim'], bg=COLORS['panel'],
                                    width=8, anchor='w')
            stats_label.pack(side='left')

            # Agent name on right
            agent_label = tk.Label(tool_row, text="", font=('Consolas', 11),
                                    fg=COLORS['text_dim'], bg=COLORS['panel'],
                                    anchor='e')
            agent_label.pack(side='right')

            self.tool_labels[tool] = name_label
            self.tool_stats[tool] = stats_label
            self.tool_agents[tool] = agent_label

    def _build_tools_panel(self):
        """ML Tools status panel (legacy - not used)."""
        panel = tk.LabelFrame(self.main_frame, text=" ML TOOLS ", font=('Consolas', 10, 'bold'),
                              fg=COLORS['green'], bg=COLORS['panel'], bd=1,
                              relief='solid', highlightbackground=COLORS['border'])
        panel.pack(fill='both', expand=True)

        inner = tk.Frame(panel, bg=COLORS['panel'], padx=10, pady=8)
        inner.pack(fill='both', expand=True)

        # Tool indicators
        self.tool_labels = {}
        self.tool_agents = {}

        tools_frame = tk.Frame(inner, bg=COLORS['panel'])
        tools_frame.pack(fill='x')

        # Create 2 columns
        left_col = tk.Frame(tools_frame, bg=COLORS['panel'])
        left_col.pack(side='left', fill='both', expand=True)
        right_col = tk.Frame(tools_frame, bg=COLORS['panel'])
        right_col.pack(side='right', fill='both', expand=True)

        for i, tool in enumerate(ML_TOOLS):
            col = left_col if i < 4 else right_col

            tool_row = tk.Frame(col, bg=COLORS['panel'])
            tool_row.pack(fill='x', pady=2)

            dot = tk.Label(tool_row, text="●", font=('Consolas', 11),
                          fg=COLORS['text_dim'], bg=COLORS['panel'])
            dot.pack(side='left')

            name = tk.Label(tool_row, text=tool, font=('Consolas', 11),
                           fg=COLORS['text_dim'], bg=COLORS['panel'], width=10, anchor='w')
            name.pack(side='left', padx=(3, 0))

            agent = tk.Label(tool_row, text="", font=('Consolas', 11),
                            fg=COLORS['text_dim'], bg=COLORS['panel'])
            agent.pack(side='right')

            self.tool_labels[tool] = (dot, name)
            self.tool_agents[tool] = agent

    def _update_loop(self):
        """Background update loop."""
        while self.running:
            try:
                # Get GPU stats
                gpu_stats = self.gpu.get_stats()

                # Get system stats (CPU/RAM)
                sys_stats = SystemStats.get_stats()

                # Read state files
                tool_states = self._read_state_file(STATE_FILE)
                vbus_state = self._read_state_file(VBUS_FILE)

                # Schedule UI update on main thread
                self.after(0, lambda g=gpu_stats, s=sys_stats, t=tool_states, v=vbus_state:
                          self._update_ui(g, s, t, v))

            except Exception as e:
                print(f"Update error: {e}")

            time.sleep(1)  # Update every second

    def _read_state_file(self, path):
        """Read JSON state file."""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _update_ui(self, gpu_stats, sys_stats, tool_states, vbus_state):
        """Update UI with new data."""
        # Update GPU name
        self.gpu_name_label.config(text=gpu_stats.get('name', GPU_NAME))

        # Update System (CPU/RAM) panel
        cpu_pct = sys_stats.get('cpu_percent', 0)
        cpu_color = COLORS['green'] if cpu_pct < 70 else (COLORS['yellow'] if cpu_pct < 90 else COLORS['red'])
        self.cpu_label.config(text=f"{cpu_pct}%", fg=cpu_color)

        # Update cores/threads in use
        cores_active = sys_stats.get('cores_active', 0)
        threads_active = sys_stats.get('threads_active', 0)
        cores_color = COLORS['cyan'] if cores_active < CPU_CORES * 0.7 else (COLORS['yellow'] if cores_active < CPU_CORES else COLORS['red'])
        self.cores_label.config(text=f"{cores_active}/{CPU_CORES}/{threads_active}", fg=cores_color)

        # Update L3 cache usage
        l3_used = sys_stats.get('l3_cache_used_mb', 0)
        l3_pct = (l3_used / CPU_L3_CACHE_MB) * 100 if CPU_L3_CACHE_MB > 0 else 0
        l3_color = COLORS['purple'] if l3_pct < 60 else (COLORS['yellow'] if l3_pct < 85 else COLORS['red'])
        self.l3_cache_label.config(text=f"{l3_used}/{CPU_L3_CACHE_MB}MB", fg=l3_color)

        ram_used = sys_stats.get('ram_used_gb', 0)
        ram_free = sys_stats.get('ram_free_gb', RAM_TOTAL_GB)
        ram_color = COLORS['green'] if ram_free > RAM_WARNING_GB else (COLORS['yellow'] if ram_free > RAM_CRITICAL_GB else COLORS['red'])

        # Only update RAM labels if they exist (removed from System panel)
        if self.ram_used_label:
            self.ram_used_label.config(text=f"{ram_used:.0f}GB", fg=COLORS['text'])
        if self.ram_free_label:
            self.ram_free_label.config(text=f"{ram_free:.0f}GB", fg=ram_color)

        # Update temperature
        temp = gpu_stats.get('temperature', 0)
        temp_color = COLORS['green'] if temp < 70 else (COLORS['yellow'] if temp < 85 else COLORS['red'])
        self.temp_label.config(text=f"{temp}°C", fg=temp_color)

        # Update utilization
        util = gpu_stats.get('utilization', 0)
        util_color = COLORS['green'] if util < 80 else (COLORS['yellow'] if util < 95 else COLORS['red'])
        self.util_label.config(text=f"{util}%", fg=util_color)

        # Update power
        power = gpu_stats.get('power_draw', 0)
        self.power_label.config(text=f"{power}W")

        # Update VRAM
        vram_used = gpu_stats.get('vram_used_mb', 0)
        vram_total = gpu_stats.get('vram_total_mb', GPU_VRAM_TOTAL_MB)
        vram_pct = gpu_stats.get('vram_percent', 0)

        self.vram_label.config(text=f"{vram_used:,} / {vram_total:,} MB")

        vram_color = COLORS['green'] if vram_pct < 70 else (COLORS['yellow'] if vram_pct < 90 else COLORS['red'])
        self.vram_percent_label.config(text=f"{vram_pct:.1f}%", fg=vram_color)

        # Draw VRAM gauge
        self._draw_gauge(self.vram_canvas, vram_pct, vram_color)

        # Update status dot
        if gpu_stats.get('real_data'):
            self.status_dot.config(fg=COLORS['green'])
        else:
            self.status_dot.config(fg=COLORS['yellow'])

        # Get actual VBUS cache stats from Resource Protector
        vbus_cache = self._get_vbus_cache_stats()
        vbus_active = vbus_cache.get('active', False)
        vbus_hits = vbus_cache.get('hits', 0)
        prot_level = vbus_cache.get('protection_level', 'normal')

        # Update VBUS ACTIVE/INACTIVE indicator
        if vbus_active:
            self.vbus_active_label.config(text="ACTIVE", fg=COLORS['green'])
        else:
            self.vbus_active_label.config(text="INACTIVE", fg=COLORS['text_dim'])

        self.vbus_hits.config(text=f"{vbus_hits:,} hits")

        # Get allocation values against TOTAL resources
        l1_alloc_gb = vbus_cache.get('l1_allocated_gb', 0)
        l1_total_gb = vbus_cache.get('l1_total_gb', GPU_VRAM_TOTAL_GB)  # 12 GB
        l2_alloc_gb = vbus_cache.get('l2_allocated_gb', 0)
        l2_total_gb = vbus_cache.get('l2_total_gb', RAM_TOTAL_GB)  # 128 GB
        l3_alloc_gb = vbus_cache.get('l3_allocated_gb', 0)
        l3_total_tb = vbus_cache.get('l3_total_tb', SSD_TOTAL_TB)  # 4 TB

        # Calculate percentages for bars (against total available)
        l1_pct = (l1_alloc_gb / l1_total_gb * 100) if l1_total_gb > 0 else 0
        l2_pct = (l2_alloc_gb / l2_total_gb * 100) if l2_total_gb > 0 else 0
        l3_pct = (l3_alloc_gb / (l3_total_tb * 1024) * 100) if l3_total_tb > 0 else 0  # Convert TB to GB

        # Flash warning for critical VRAM
        vram_free = vram_total - vram_used
        if vram_free < VRAM_CRITICAL or prot_level == 'emergency':
            # Critical - flash red
            self.kill_btn.config(bg='#ff0000')
            self.configure(bg='#2a0a0a')
        elif vram_free < VRAM_WARNING or prot_level == 'critical':
            # Warning - yellow tint
            self.kill_btn.config(bg=COLORS['orange'])
            self.configure(bg='#1a1a0a')
        elif prot_level == 'warning':
            self.kill_btn.config(bg=COLORS['yellow'])
            self.configure(bg=COLORS['bg'])
        else:
            # Normal
            self.kill_btn.config(bg=COLORS['red'])
            self.configure(bg=COLORS['bg'])

        # Draw cache bars using percentages (color based on usage)
        l1_color = COLORS['cyan'] if l1_pct < 50 else (COLORS['yellow'] if l1_pct < 80 else COLORS['red'])
        l2_color = COLORS['yellow'] if l2_pct < 50 else (COLORS['orange'] if l2_pct < 80 else COLORS['red'])
        l3_color = COLORS['text_dim'] if l3_pct < 50 else (COLORS['yellow'] if l3_pct < 80 else COLORS['red'])

        self._draw_cache_bar(self.l1_bar, l1_pct, 100, l1_color)
        self._draw_cache_bar(self.l2_bar, l2_pct, 100, l2_color)
        self._draw_cache_bar(self.l3_bar, l3_pct, 100, l3_color)

        # Get active usage (entries currently being accessed)
        l1_active_gb = vbus_cache.get('l1_active_gb', 0)
        l2_active_gb = vbus_cache.get('l2_active_gb', 0)
        l3_active_gb = vbus_cache.get('l3_active_gb', 0)

        # Update labels: active / allocated / max
        self.l1_label.config(text=f"{l1_active_gb:.1f} / {l1_alloc_gb:.1f} / {l1_total_gb:.0f}", fg=l1_color)
        self.l2_label.config(text=f"{l2_active_gb:.1f} / {l2_alloc_gb:.1f} / {l2_total_gb:.0f}", fg=l2_color)
        # L3 in TB
        l3_alloc_tb = l3_alloc_gb / 1024.0
        l3_active_tb = l3_active_gb / 1024.0
        self.l3_label.config(text=f"{l3_active_tb:.2f} / {l3_alloc_tb:.2f} / {l3_total_tb:.0f}", fg=l3_color)

        # Update ML tools with real data - show status and agent name
        ml_stats = self._get_ml_tools_stats()
        active_count = 0

        for tool in ML_TOOLS:
            tool_data = ml_stats.get(tool, {})
            is_active = tool_data.get('active', False)
            agent_name = tool_data.get('agent', '')

            if is_active:
                active_count += 1
                # Tool name bright when active
                if tool in self.tool_labels:
                    self.tool_labels[tool].config(fg=COLORS['green'])
                # Status column shows ACTIVE
                if tool in self.tool_stats:
                    self.tool_stats[tool].config(text="ACTIVE", fg=COLORS['green'])
                # Agent column shows agent name
                if tool in self.tool_agents:
                    agent_color = AGENT_COLORS.get(agent_name, COLORS['cyan'])
                    self.tool_agents[tool].config(text=agent_name, fg=agent_color)
            else:
                if tool in self.tool_labels:
                    self.tool_labels[tool].config(fg=COLORS['text'])
                if tool in self.tool_stats:
                    self.tool_stats[tool].config(text="idle", fg=COLORS['text_dim'])
                if tool in self.tool_agents:
                    self.tool_agents[tool].config(text="", fg=COLORS['text_dim'])

        # Update ML ACTIVE/INACTIVE indicator
        if active_count > 0:
            self.ml_active_label.config(text="ACTIVE", fg=COLORS['green'])
            self.ml_total_label.config(text=f"{active_count} in use", fg=COLORS['green'])
        else:
            self.ml_active_label.config(text="INACTIVE", fg=COLORS['text_dim'])
            self.ml_total_label.config(text="0 in use", fg=COLORS['text_dim'])

        # Update footer with real session timer and GPU activity
        session_duration = datetime.now() - self.session_start_time
        hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        session_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Determine if GPU is really active (utilization > 5% means actual work)
        gpu_util = gpu_stats.get('utilization', 0)
        self.gpu_is_active = gpu_util > 5 and gpu_stats.get('real_data', False)

        if gpu_stats.get('real_data'):
            if self.gpu_is_active:
                # Bright green when GPU is actually working
                self.footer.config(
                    text=f"Live GPU data | Session: {session_str} | {datetime.now().strftime('%H:%M:%S')}",
                    fg=COLORS['green']
                )
            else:
                # Normal text when GPU connected but idle
                self.footer.config(
                    text=f"Live GPU data | Session: {session_str} | {datetime.now().strftime('%H:%M:%S')}",
                    fg=COLORS['text_dim']
                )
        else:
            self.footer.config(
                text=f"No GPU driver | Session: {session_str} | {datetime.now().strftime('%H:%M:%S')}",
                fg=COLORS['text_dim']
            )

    def _draw_gauge(self, canvas, percent, color):
        """Draw a horizontal gauge bar."""
        canvas.delete('all')
        width = canvas.winfo_width() - 4
        height = canvas.winfo_height() - 4

        if width <= 0:
            width = 320

        # Background
        canvas.create_rectangle(2, 2, width + 2, height + 2, fill=COLORS['bg'], outline='')

        # Fill
        fill_width = int((percent / 100) * width)
        if fill_width > 0:
            canvas.create_rectangle(2, 2, fill_width + 2, height + 2, fill=color, outline='')

        # Threshold markers
        mark_70 = int(0.70 * width) + 2
        mark_90 = int(0.90 * width) + 2
        canvas.create_line(mark_70, 2, mark_70, height + 2, fill=COLORS['yellow'], dash=(2, 2))
        canvas.create_line(mark_90, 2, mark_90, height + 2, fill=COLORS['red'], dash=(2, 2))

    def _draw_cache_bar(self, canvas, used, total, color):
        """Draw a cache tier bar."""
        canvas.delete('all')
        width = 178
        height = 10

        # Background
        canvas.create_rectangle(1, 1, width, height, fill=COLORS['bg'], outline='')

        # Fill
        if total > 0:
            fill_width = int((used / total) * width)
            if fill_width > 0:
                canvas.create_rectangle(1, 1, fill_width, height, fill=color, outline='')

    def _emergency_kill(self):
        """Trigger emergency VRAM kill - multiple methods for reliability."""
        self.kill_btn.config(state='disabled', text="...")
        self.footer.config(text="Killing GPU processes...", fg=COLORS['yellow'])

        def do_kill():
            success = False
            error_msg = None

            try:
                # Method 1: Kill all GPU compute processes via nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                if result.returncode == 0 and result.stdout.strip():
                    pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
                    for pid in pids:
                        try:
                            if os.name == 'nt':
                                subprocess.run(['taskkill', '/F', '/PID', pid],
                                              capture_output=True, timeout=5,
                                              creationflags=subprocess.CREATE_NO_WINDOW)
                            else:
                                subprocess.run(['kill', '-9', pid], capture_output=True, timeout=5)
                        except:
                            pass
                    success = True

                # Method 2: WSL cupy memory clear (for RAPIDS processes)
                wsl_cmd = '''
source /home/oy6/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate rapids-24.12 2>/dev/null
python -c "
import cupy as cp
import gc
try:
    mempool = cp.get_default_memory_pool()
    pinned = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned.free_all_blocks()
    gc.collect()
    print('CLEARED')
except: pass
" 2>/dev/null
'''
                try:
                    result = subprocess.run(
                        ['wsl', '-d', 'Ubuntu', '-e', 'bash', '-c', wsl_cmd],
                        capture_output=True, text=True, timeout=15
                    )
                    if 'CLEARED' in result.stdout:
                        success = True
                except:
                    pass

                # Method 3: nvidia-smi GPU reset (last resort)
                if not success:
                    try:
                        subprocess.run(
                            ['nvidia-smi', '--gpu-reset', '-i', '0'],
                            capture_output=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                        )
                        success = True
                    except:
                        pass

            except Exception as e:
                error_msg = str(e)

            self.after(0, lambda: self._kill_complete(success, error_msg))

        threading.Thread(target=do_kill, daemon=True).start()

    def _kill_complete(self, success, error=None):
        """Handle kill completion."""
        self.kill_btn.config(state='normal', text="KILL")
        if success:
            self.footer.config(text="VRAM cleared successfully", fg=COLORS['green'])
        else:
            self.footer.config(text=f"Kill failed: {error or 'unknown'}", fg=COLORS['red'])

    def _get_vbus_cache_stats(self):
        """Read VBUS state - only shows ACTIVE when VBUS is actually running."""
        stats = {
            'active': False,
            'protection_level': 'normal',
            'l1_allocated_gb': 0.0,
            'l1_active_gb': 0.0,
            'l1_total_gb': GPU_VRAM_TOTAL_GB,
            'l1_entries': 0,
            'l2_allocated_gb': 0.0,
            'l2_active_gb': 0.0,
            'l2_total_gb': RAM_TOTAL_GB,
            'l2_entries': 0,
            'l3_allocated_gb': 0.0,
            'l3_active_gb': 0.0,
            'l3_total_tb': SSD_TOTAL_TB,
            'l3_entries': 0,
            'hits': 0,
        }

        try:
            # Source 1: VBUS Resource Protector state file
            if VBUS_STATE_FILE.exists():
                with open(VBUS_STATE_FILE, 'r') as f:
                    vbus_data = json.load(f)
                    stats['active'] = vbus_data.get('vbus_running', False)
                    stats['protection_level'] = vbus_data.get('protection_level', 'normal')
                    stats['l1_allocated_gb'] = vbus_data.get('l1_allocated_mb', 0) / 1024.0
                    stats['l1_active_gb'] = vbus_data.get('l1_active_gb', 0)
                    stats['l1_entries'] = vbus_data.get('l1_entries', 0)
                    stats['l2_allocated_gb'] = vbus_data.get('l2_allocated_gb', 0)
                    stats['l2_active_gb'] = vbus_data.get('l2_active_gb', 0)
                    stats['l2_entries'] = vbus_data.get('l2_entries', 0)
                    stats['l3_allocated_gb'] = vbus_data.get('l3_allocated_gb', 0)
                    stats['l3_active_gb'] = vbus_data.get('l3_active_gb', 0)
                    stats['l3_entries'] = vbus_data.get('l3_entries', 0)
                    stats['hits'] = vbus_data.get('total_hits', 0)

            # Source 2: MAPIE Engine VBUS status
            mapie_vbus = SCRIPT_DIR.parent / 'MAPIE Engine' / 'output' / 'vbus_status.json'
            if mapie_vbus.exists():
                with open(mapie_vbus, 'r') as f:
                    mapie_data = json.load(f)
                    if mapie_data.get('vbus_active', False):
                        stats['active'] = True
                        stats['l1_entries'] = mapie_data.get('l1_entries', stats['l1_entries'])
                        stats['l2_entries'] = mapie_data.get('l2_entries', stats['l2_entries'])

            # Source 3: Check L3 disk cache (actual files = actual cache)
            l3_path = VBUS_CACHE_DIR / 'l3'
            if l3_path.exists():
                total_size = 0
                file_count = 0
                for f in l3_path.rglob('*'):
                    if f.is_file():
                        total_size += f.stat().st_size
                        file_count += 1
                if file_count > 0:
                    stats['l3_allocated_gb'] = total_size / (1024**3)
                    stats['l3_entries'] = file_count

        except Exception:
            pass

        return stats

    def _get_ml_tools_stats(self):
        """Get ML tool usage - reads from agent state file only (no inference)."""
        stats = {tool: {'active': False, 'agent': ''} for tool in ML_TOOLS}

        try:
            # Read from ML tools state file written by agents
            if ML_TOOLS_STATE_FILE.exists():
                with open(ML_TOOLS_STATE_FILE, 'r') as f:
                    ml_data = json.load(f)
                    for tool in ML_TOOLS:
                        if tool in ml_data:
                            stats[tool]['active'] = ml_data[tool].get('active', False)
                            stats[tool]['agent'] = ml_data[tool].get('agent', '')
        except Exception:
            pass

        return stats

    def _on_close(self):
        """Handle window close."""
        self.running = False
        self.gpu.shutdown()
        self.destroy()


def main():
    """Main entry point."""
    app = ViewerDBXMonitor()
    app.mainloop()


if __name__ == '__main__':
    main()
