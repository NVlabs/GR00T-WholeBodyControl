# Ранбук: VLA

Левая рука, `left_run1/checkpoint-55000`.

---
## 1. Порядок запуска

### A100 (в контейнере, одной строкой)

```bash
ssh root@100.64.0.21
docker exec -it tactile-train bash
source /opt/Isaac-GR00T/.venv/bin/activate && cd /opt/Isaac-GR00T
pkill -f run_gr00t_server; sleep 3
python gr00t/eval/run_gr00t_server.py --model-path /data/checkpoints/left_run1/checkpoint-55000 --embodiment-tag NEW_EMBODIMENT --port 5555
```

### A100, хост, в tmux

```bash
tmux new -s socat
socat TCP-LISTEN:5555,fork,reuseaddr TCP:172.17.0.2:5555
```

### Робот, по порядку

```bash
# 1. камера
cd ~/GR00T-WholeBodyControl && source .venv_camera/bin/activate
python -m gear_sonic.camera.composed_camera --ego-view-camera usb --ego-view-device-id 4 --port 5555

# 2. кисти — РОВНО ОДИН экземпляр
cd /home/unitree/brainco_hand_service && sudo ./bin/brainco_hand_server -n wlxfc23cd997021

# 3. WBC — ждать «Init done»
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real

# 4. паблишер клавиш
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
python3 ~/keypress.py

# 5. мост — команды в §3

### База (VLA одна)

cd ~/GR00T-WholeBodyControl && source .venv_data_collection/bin/activate && AGENT_HAND=left AGENT_ENABLE=0 SLEW_ENABLE=1 PROBE_HAND=left PROBE_ENABLE=1 HAND_ENABLE=1 PYTHONPATH=/home/unitree/unitree_sdk2_python python gear_sonic/scripts/run_vla_inference_liza.py --host 100.64.0.21 --port 5555 --camera-host localhost --camera-port 5555 --state-zmq-host localhost --state-zmq-port 5557 --action-zmq-host localhost --action-zmq-port 5556 --embodiment-tag NEW_EMBODIMENT --prompt "pick up the object and place it down" 2>&1 | tee ~/bridge_base_$(date +%Y%m%d_%H%M).log


**Клавиши:** `k` старт → `i` начальная поза (обязательно) → `p` снять паузу (тумблер) → `g` заморозка тела.


