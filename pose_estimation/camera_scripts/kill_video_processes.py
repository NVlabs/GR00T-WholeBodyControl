#!/usr/bin/env python3
import subprocess, glob, re

def find_video_nodes():
    return glob.glob("/dev/video*")

def fuser_pids(device):
    """Return set of PIDs using a video device via fuser -v"""
    try:
        out = subprocess.run(
            ["sudo", "fuser", "-v", device],
            capture_output=True, text=True, check=False
        )
        # extract PIDs
        return {int(pid) for pid in re.findall(r"\b\d+\b", out.stdout + out.stderr)}
    except Exception as e:
        print(f"Error running fuser on {device}: {e}")
        return set()

def kill_pids(pids):
    if not pids:
        return
    try:
        subprocess.run(["sudo", "kill", "-9"] + [str(pid) for pid in pids], check=False)
        print(f"Killed: {', '.join(map(str, pids))}")
    except Exception as e:
        print(f"Error killing {pids}: {e}")

def kill_video():
    nodes = find_video_nodes()
    if not nodes:
        print("No /dev/video* devices found")
        return
    print("Found video nodes:", nodes)

    all_pids = set()
    for node in nodes:
        pids = fuser_pids(node)
        if pids:
            print(f"{node} in use by: {pids}")
            all_pids |= pids

    if not all_pids:
        print("No processes to kill.")
    else:
        # Kill the processes
        kill_pids(all_pids)

def restore_video_service():
    subprocess.run(
        ["sudo", "chmod", "+x", "/unitree/module/video_hub_pc4/videohub_pc4"],
        check=False
    )

def kill_video_service():
    subprocess.run(
        ["sudo", "chmod", "-x", "/unitree/module/video_hub_pc4/videohub_pc4"],
        check=False
    )

if __name__ == "__main__":
    kill_video()
