# AI ML VBUS Dashboard - Widget Guide

## Overview
Real-time GPU monitoring widget for ViewerDBX ML operations.
Always-on-top floating panel with blue glow border.

---

## Panels

### GPU Panel
| Metric | Description |
|--------|-------------|
| TEMP | GPU core temperature (Celsius) |
| UTIL | GPU utilization percentage |
| POWER | Current power draw (Watts) |

### VRAM Panel
Shows GPU memory usage with visual gauge.
- Green: <70% usage (safe)
- Yellow: 70-90% (warning)
- Red: >90% (critical)

### SYSTEM Panel
| Metric | Format | Description |
|--------|--------|-------------|
| LOAD | XX% | CPU utilization |
| CORES | active/total/threads | Cores in use |
| L3 | used/total MB | L3 cache usage |

### VBUS PROTECTED CACHE Panel
Three-tier cache system for GPU data.

| Tier | Storage | Format |
|------|---------|--------|
| L1 | GPU VRAM | active / allocated / 12 GB max |
| L2 | System RAM | active / allocated / 128 GB max |
| L3 | SSD Disk | active / allocated / 4 TB max |

**Numbers explained:**
- **Active**: Currently being accessed
- **Allocated**: Reserved for VBUS cache
- **Max**: Total available capacity

**Status:**
- ACTIVE: VBUS cache is running (agents using GPU)
- INACTIVE: No GPU agents currently running

### ML TOOLS Panel
Shows which ML libraries are in use.

| Tool | Description |
|------|-------------|
| cuDF | GPU DataFrame (RAPIDS) |
| XGBoost | Gradient boosting |
| cuML | GPU machine learning |
| Dask-RAPIDS | Distributed GPU |
| cuGraph | GPU graph analytics |
| CatBoost | Categorical boosting |
| PyTorch | Deep learning |
| RandomForest | Ensemble learning |
| cuFFT | GPU FFT |
| cuSOLVER | GPU linear algebra |
| cuBLAS | GPU matrix ops |

When active, shows the **agent name** using that tool (e.g., "Fiona", "ProQ").

---

## Controls

### Buttons
| Button | Action |
|--------|--------|
| X | Close widget |
| KILL | Emergency GPU process kill |

### Right-Click Menu
| Option | Action |
|--------|--------|
| README / Help | Opens this file |
| Refresh Now | Force immediate data refresh |
| Clear VBUS Cache | Reset all cache tiers |
| Reset Session Timer | Restart session clock |
| Copy Stats to Clipboard | Copy current stats |
| Emergency GPU Kill | Kill all GPU processes |
| Toggle Always On Top | Pin/unpin window |
| Collapse/Expand Panels | Show/hide details |
| Close Dashboard | Exit widget |

### Dragging
Click anywhere on the widget and drag to move it.

---

## Data Sources

| Data | Source |
|------|--------|
| GPU stats | nvidia-smi / pynvml |
| VBUS state | .claude/.vbus_state.json |
| ML tools | .claude/.ml_tools_state.json |
| Agent state | .claude/.agent_state.json |

---

## Agent Status Reporting

Agents report their status using:
```bash
python widget_notify.py agent-start "AgentName"
python widget_notify.py agent-stop "AgentName"
```

When a GPU agent starts, VBUS activates and their ML tools light up.

### GPU Agents (activate VBUS)
Fiona, ProQ, JamesHunt, Sally, Randy, Madge, Berrin, Warren, Carla, Ted, Vera, Tess, Fred, Monty, Oscar

### Non-GPU Agents
Sid, Tony, Andy, Piers, Alex, Dan, Whip

---

## Color Coding

| Color | Meaning |
|-------|---------|
| Green | Good / Active / Safe |
| Yellow | Warning / Elevated |
| Orange | High / Caution |
| Red | Critical / Emergency |
| Cyan | L1 VRAM tier |
| Purple | Cache / Special |
| Gray | Inactive / Idle |

---

## Thresholds

| Resource | Warning | Critical |
|----------|---------|----------|
| VRAM Free | <1000 MB | <500 MB |
| Temperature | >70C | >85C |
| GPU Util | >80% | >95% |
| RAM Free | <16 GB | <8 GB |

---

## Files

| File | Purpose |
|------|---------|
| ViewerDBX_GPU_Monitor.pyw | Widget application |
| widget_notify.py | Agent status reporter |
| .vbus_state.json | VBUS cache state |
| .ml_tools_state.json | ML tool usage |
| .agent_state.json | Agent activity |
| WIDGET_README.md | This file |

---

## Troubleshooting

**Widget shows "No GPU driver":**
- Ensure NVIDIA drivers are installed
- Run `nvidia-smi` in terminal to verify

**VBUS shows INACTIVE:**
- No GPU agent is currently running
- Start an agent: `python widget_notify.py agent-start "Fiona"`

**ML Tools all idle:**
- No active GPU operations
- Tools activate when agents perform GPU work

**Cannot drag widget:**
- Click on dark area (not on buttons/panels)
- Drag to move

---

(c) Framecore Inc - ViewerDBX Project
