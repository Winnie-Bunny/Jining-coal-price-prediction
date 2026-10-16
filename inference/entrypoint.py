"""Prepare the mounted state directory, then drop privileges before running code."""
import os
from pathlib import Path
import sys

state = Path(os.environ.get('STATE_DIR', '/app/results'))
state.mkdir(parents=True, exist_ok=True)
if os.getuid() == 0:
    os.chown(state, 10001, 10001)
    os.setgroups([])
    os.setgid(10001)
    os.setuid(10001)
os.execvp(sys.argv[1], sys.argv[1:])
