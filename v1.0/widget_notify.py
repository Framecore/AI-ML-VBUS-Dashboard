#!/usr/bin/env python3
"""
Widget Notify - Agent Status Reporter for AI ML VBUS Dashboard
===============================================================

Agents call this script to report their status to the widget.
The widget reads from the state files this script writes to.

Usage:
    python widget_notify.py agent-start "AgentName"
    python widget_notify.py agent-stop "AgentName"
    python widget_notify.py vbus-start
    python widget_notify.py vbus-stop
    python widget_notify.py ml-tool-active "cuDF"
    python widget_notify.py ml-tool-inactive "cuDF"
    python widget_notify.py cache-update L1 64 512
    python widget_notify.py status
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# State files in Widget Engine/state folder
SCRIPT_DIR = Path(__file__).parent
STATE_DIR = SCRIPT_DIR.parent / 'state'
VBUS_STATE_FILE = STATE_DIR / '.vbus_state.json'
ML_TOOLS_STATE_FILE = STATE_DIR / '.ml_tools_state.json'
AGENT_STATE_FILE = STATE_DIR / '.agent_state.json'

# Valid ML tools (matching widget)
ML_TOOLS = [
    'cuDF', 'XGBoost', 'cuML', 'Dask-RAPIDS', 'cuGraph',
    'CatBoost', 'PyTorch', 'RandomForest', 'cuFFT', 'cuSOLVER', 'cuBLAS'
]

# GPU agents (short names as used in agent .md files)
GPU_AGENTS = [
    'Berrin', 'Fiona', 'James Hunt', 'ProQ', 'Madge', 'Sally',
    'Randy', 'Warren', 'Carla', 'Ted', 'Vera', 'Tess', 'Fred', 'Monty', 'Oscar'
]
NON_GPU_AGENTS = ['Sid', 'Tony', 'Andy', 'Piers', 'Alex', 'Dan', 'Whip']

# Agent-to-tools mapping
AGENT_TOOLS = {
    'Fiona': ['cuDF', 'cuML', 'XGBoost', 'CatBoost', 'PyTorch', 'cuBLAS', 'cuSOLVER'],
    'ProQ': ['cuDF', 'cuML', 'XGBoost', 'cuGraph', 'Dask-RAPIDS', 'cuFFT', 'RandomForest'],
    'James Hunt': ['cuDF', 'cuML', 'XGBoost', 'cuGraph', 'Dask-RAPIDS', 'cuBLAS'],
    'Sally': ['cuDF', 'cuML', 'XGBoost'],
    'Randy': ['cuDF'],
    'Madge': ['cuDF'],
    'Berrin': ['cuDF', 'cuML'],
    'Warren': ['cuDF'],
    'Carla': ['cuDF', 'cuML'],
    'Ted': ['cuDF', 'cuGraph'],
    'Vera': ['cuDF'],
    'Tess': ['cuDF', 'cuML'],
    'Fred': ['cuDF', 'XGBoost'],
    'Monty': ['cuDF'],
    'Oscar': ['cuDF'],
}


def load_json(filepath):
    try:
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}


def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_timestamp():
    return datetime.now().isoformat()


def agent_start(agent_name):
    timestamp = get_timestamp()

    # Update agent state
    agents = load_json(AGENT_STATE_FILE)
    agents[agent_name] = {'active': True, 'started': timestamp, 'gpu': agent_name in GPU_AGENTS}
    save_json(AGENT_STATE_FILE, agents)

    # GPU agent activates VBUS and allocates cache
    if agent_name in GPU_AGENTS:
        vbus = load_json(VBUS_STATE_FILE)
        vbus['vbus_running'] = True
        vbus['active_agent'] = agent_name
        vbus['last_updated'] = timestamp

        # Allocate cache based on agent's tool count
        tool_count = len(AGENT_TOOLS.get(agent_name, []))

        # L1 (VRAM): ~0.5 GB per tool, active = 80% of allocated
        l1_alloc = tool_count * 0.5
        vbus['l1_allocated_mb'] = vbus.get('l1_allocated_mb', 0) + (l1_alloc * 1024)
        vbus['l1_active_gb'] = vbus.get('l1_active_gb', 0) + (l1_alloc * 0.8)

        # L2 (RAM): ~1 GB per tool for staging
        l2_alloc = tool_count * 1.0
        vbus['l2_allocated_gb'] = vbus.get('l2_allocated_gb', 0) + l2_alloc
        vbus['l2_active_gb'] = vbus.get('l2_active_gb', 0) + (l2_alloc * 0.5)

        # L3 (SSD): ~2 GB per tool for cold storage
        l3_alloc = tool_count * 2.0
        vbus['l3_allocated_gb'] = vbus.get('l3_allocated_gb', 0) + l3_alloc
        vbus['l3_active_gb'] = vbus.get('l3_active_gb', 0) + (l3_alloc * 0.2)

        # Track entries
        vbus['l1_entries'] = vbus.get('l1_entries', 0) + tool_count
        vbus['l2_entries'] = vbus.get('l2_entries', 0) + tool_count
        vbus['l3_entries'] = vbus.get('l3_entries', 0) + tool_count

        save_json(VBUS_STATE_FILE, vbus)

    # Activate ML tools for this agent
    if agent_name in AGENT_TOOLS:
        ml = load_json(ML_TOOLS_STATE_FILE)
        for tool in AGENT_TOOLS[agent_name]:
            ml[tool] = {'active': True, 'started': timestamp, 'agent': agent_name}
        save_json(ML_TOOLS_STATE_FILE, ml)
        print(f"Agent {agent_name} started with: {', '.join(AGENT_TOOLS[agent_name])}")
    else:
        print(f"Agent {agent_name} started")


def agent_stop(agent_name):
    timestamp = get_timestamp()

    agents = load_json(AGENT_STATE_FILE)
    if agent_name in agents:
        agents[agent_name]['active'] = False
        agents[agent_name]['stopped'] = timestamp
    save_json(AGENT_STATE_FILE, agents)

    # Deallocate cache and check if any GPU agent still active
    vbus = load_json(VBUS_STATE_FILE)

    # Deallocate this agent's cache
    if agent_name in GPU_AGENTS:
        tool_count = len(AGENT_TOOLS.get(agent_name, []))

        # Remove L1 allocation
        l1_alloc = tool_count * 0.5
        vbus['l1_allocated_mb'] = max(0, vbus.get('l1_allocated_mb', 0) - (l1_alloc * 1024))
        vbus['l1_active_gb'] = max(0, vbus.get('l1_active_gb', 0) - (l1_alloc * 0.8))

        # Remove L2 allocation
        l2_alloc = tool_count * 1.0
        vbus['l2_allocated_gb'] = max(0, vbus.get('l2_allocated_gb', 0) - l2_alloc)
        vbus['l2_active_gb'] = max(0, vbus.get('l2_active_gb', 0) - (l2_alloc * 0.5))

        # Remove L3 allocation
        l3_alloc = tool_count * 2.0
        vbus['l3_allocated_gb'] = max(0, vbus.get('l3_allocated_gb', 0) - l3_alloc)
        vbus['l3_active_gb'] = max(0, vbus.get('l3_active_gb', 0) - (l3_alloc * 0.2))

        # Remove entries
        vbus['l1_entries'] = max(0, vbus.get('l1_entries', 0) - tool_count)
        vbus['l2_entries'] = max(0, vbus.get('l2_entries', 0) - tool_count)
        vbus['l3_entries'] = max(0, vbus.get('l3_entries', 0) - tool_count)

    if vbus.get('active_agent') == agent_name:
        vbus['active_agent'] = None
        any_active = any(d.get('active') and a in GPU_AGENTS for a, d in agents.items())
        if not any_active:
            vbus['vbus_running'] = False
            # Reset all allocations when no GPU agents active
            vbus['l1_allocated_mb'] = 0
            vbus['l1_active_gb'] = 0
            vbus['l2_allocated_gb'] = 0
            vbus['l2_active_gb'] = 0
            vbus['l3_allocated_gb'] = 0
            vbus['l3_active_gb'] = 0
            vbus['l1_entries'] = 0
            vbus['l2_entries'] = 0
            vbus['l3_entries'] = 0

    vbus['last_updated'] = timestamp
    save_json(VBUS_STATE_FILE, vbus)

    # Deactivate agent's ML tools
    if agent_name in AGENT_TOOLS:
        ml = load_json(ML_TOOLS_STATE_FILE)
        for tool in AGENT_TOOLS[agent_name]:
            if tool in ml and ml[tool].get('agent') == agent_name:
                ml[tool]['active'] = False
                ml[tool]['stopped'] = timestamp
        save_json(ML_TOOLS_STATE_FILE, ml)

    print(f"Agent {agent_name} stopped")


def vbus_start():
    vbus = load_json(VBUS_STATE_FILE)
    vbus['vbus_running'] = True
    vbus['last_updated'] = get_timestamp()
    save_json(VBUS_STATE_FILE, vbus)
    print("VBUS started")


def vbus_stop():
    vbus = load_json(VBUS_STATE_FILE)
    vbus['vbus_running'] = False
    vbus['last_updated'] = get_timestamp()
    save_json(VBUS_STATE_FILE, vbus)
    print("VBUS stopped")


def ml_tool_active(tool_name):
    ml = load_json(ML_TOOLS_STATE_FILE)
    ml[tool_name] = {'active': True, 'started': get_timestamp()}
    save_json(ML_TOOLS_STATE_FILE, ml)
    print(f"ML tool {tool_name} active")


def ml_tool_inactive(tool_name):
    ml = load_json(ML_TOOLS_STATE_FILE)
    if tool_name in ml:
        ml[tool_name]['active'] = False
        ml[tool_name]['stopped'] = get_timestamp()
    save_json(ML_TOOLS_STATE_FILE, ml)
    print(f"ML tool {tool_name} inactive")


def cache_update(level, entries, size_mb):
    vbus = load_json(VBUS_STATE_FILE)
    level = level.upper()
    if level == 'L1':
        vbus['l1_entries'] = entries
        vbus['l1_allocated_mb'] = size_mb
    elif level == 'L2':
        vbus['l2_entries'] = entries
        vbus['l2_allocated_gb'] = size_mb / 1024
    elif level == 'L3':
        vbus['l3_entries'] = entries
        vbus['l3_allocated_gb'] = size_mb / 1024
    vbus['last_updated'] = get_timestamp()
    save_json(VBUS_STATE_FILE, vbus)
    print(f"Cache {level}: {entries} entries, {size_mb}MB")


def show_status():
    print("=== VBUS ===")
    for k, v in load_json(VBUS_STATE_FILE).items():
        print(f"  {k}: {v}")
    print("\n=== ML Tools ===")
    for k, v in load_json(ML_TOOLS_STATE_FILE).items():
        print(f"  {k}: {v}")
    print("\n=== Agents ===")
    for k, v in load_json(AGENT_STATE_FILE).items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == 'agent-start' and len(sys.argv) >= 3:
        agent_start(sys.argv[2])
    elif cmd == 'agent-stop' and len(sys.argv) >= 3:
        agent_stop(sys.argv[2])
    elif cmd == 'vbus-start':
        vbus_start()
    elif cmd == 'vbus-stop':
        vbus_stop()
    elif cmd == 'ml-tool-active' and len(sys.argv) >= 3:
        ml_tool_active(sys.argv[2])
    elif cmd == 'ml-tool-inactive' and len(sys.argv) >= 3:
        ml_tool_inactive(sys.argv[2])
    elif cmd == 'cache-update' and len(sys.argv) >= 5:
        cache_update(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    elif cmd == 'status':
        show_status()
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)
