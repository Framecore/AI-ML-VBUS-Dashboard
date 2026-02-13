# Widget Engine Version Control

## Current Version: v1.0 (LOCKED)

## Version Policy

### LOCKED Versions
Once a version is marked as LOCKED, it **CANNOT** be modified:
- No edits to any files
- No deletions
- No additions
- Files are set to read-only

### Creating New Versions
To create a new version (e.g., v2.0):
1. Create new directory: `Widget Engine/v2.0/`
2. Copy files FROM the locked version (do not move)
3. Make modifications in the new version only
4. Document changes in this file

### Version History

| Version | Status | Date | Notes |
|---------|--------|------|-------|
| v1.0 | LOCKED | 2026-02-13 | Initial release - AI ML VBUS Dashboard |

## Directory Structure

```
Widget Engine/
├── VERSION.md          (this file)
├── state/              (runtime state files - shared by all versions)
│   ├── .vbus_state.json
│   ├── .ml_tools_state.json
│   └── .agent_state.json
├── v1.0/               (LOCKED - read-only)
│   ├── ViewerDBX_GPU_Monitor.pyw
│   ├── widget_notify.py
│   └── README.md
└── v2.0/               (future versions go here)
```

## GitHub Repository
https://github.com/Framecore/AI-ML-VBUS-Dashboard

---
(c) Framecore Inc - ViewerDBX Project
