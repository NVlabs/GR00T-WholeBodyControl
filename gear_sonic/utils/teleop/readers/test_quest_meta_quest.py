"""Test: verifica que Meta Quest envía poses de controladores via meta_quest_teleop."""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path (for gear_sonic if needed)
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    try:
        from meta_quest_teleop.reader import MetaQuestReader
    except ImportError as e:
        print("[ERROR] meta_quest_teleop no instalado. Ejecuta:")
        print(
            "  uv pip install --python .venv_sim/bin/python "
            "'meta_quest_teleop @ git+https://github.com/BrikHMP18/meta_quest_teleop.git@07bc15437f767c3517138367b6b3e3910b388c76'"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test Meta Quest controller data via meta_quest_teleop")
    parser.add_argument("--ip-address", type=str, default=None, help="Quest IP for WiFi (omit for USB)")
    args = parser.parse_args()
    ip = args.ip_address
    print(f"[test] Conectando a Quest (ip={ip or 'USB'})...")
    reader = MetaQuestReader(ip_address=ip, run=True)
    print("[test] MetaQuestReader iniciado. Mueve los mandos para ver datos...")
    print("[test] Ctrl+C para parar.\n")

    try:
        while True:
            left = reader.get_hand_controller_transform_ros("left")
            right = reader.get_hand_controller_transform_ros("right")
            grip_l = reader.get_grip_value("left")
            grip_r = reader.get_grip_value("right")
            trig = reader.get_trigger_value("right")
            js = reader.get_joystick_value("right")

            status = "L:OK" if left is not None else "L:--"
            status += " R:OK" if right is not None else " R:--"
            line = f"{status} | grip=({grip_l:.2f},{grip_r:.2f}) trig={trig:.2f} js={js}"
            if left is not None:
                pos_l = left[:3, 3]
                line += f" | L=({pos_l[0]:.2f},{pos_l[1]:.2f},{pos_l[2]:.2f})"
            if right is not None:
                pos_r = right[:3, 3]
                line += f" R=({pos_r[0]:.2f},{pos_r[1]:.2f},{pos_r[2]:.2f})"
            print(f"\r{line}   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[test] Parado.")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
