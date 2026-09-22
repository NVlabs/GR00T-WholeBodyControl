# Simple Custom README — BrainCo teleoperation

## tmux launchers

Используются два отдельных скрипта:

```text
gear_sonic/scripts/launch_brainco_robot_tmux.sh  # запускать на роботе
gear_sonic/scripts/launch_brainco_host_tmux.sh   # запускать на host-компьютере
```

Установить `tmux` нужно на обеих машинах:

```bash
sudo apt install tmux
```

Запуск robot stack:

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh
tmux attach -t brainco_robot
```

Запуск host stack:

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection
ROBOT_HOST=192.168.50.132 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh
tmux attach -t brainco_host
```

Default-профиль обоих launcher’ов:

- robot `192.168.50.132` отправляет RGB с `ego_view` RealSense `1280x720` и
  `head` USB `/dev/video6` `1600x896`; обе камеры работают при `15 FPS`;
- RealSense depth-потоки на роботе отключены и по сети не отправляются;
- host записывает обе RGB-камеры, локальную `/dev/video4` как
  `external-view-camera` и G1 telemetry;
- viewer показывает `ego_view`, `head` и `external-view-camera`;
- raw depth и depth preview video не сохраняются.

Если внешняя камера имеет другой путь, задайте его через
`EXTERNAL_VIEW_CAMERA_DEVICE`. Пустое значение отключает external camera.
Путь можно проверить на host командой `v4l2-ctl --list-devices`.

`--replace` останавливает сессию с тем же именем и создаёт её заново:


# Robot
```bash

cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace
tmux attach -t brainco_robot
```

# Host

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection/
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace
tmux attach -t brainco_host
```

Переключение tmux-окон: `Ctrl+b`, затем номер окна, либо `Ctrl+b`, затем `n` / `p`.
Отсоединение без остановки: `Ctrl+b`, затем `d`.

```bash
tmux attach -t brainco_robot
tmux attach -t brainco_host
tmux kill-session -t brainco_robot
tmux kill-session -t brainco_host
```

---

# Робот

## 1. Подготовка

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection

bash install_scripts/install_camera_server.sh
bash install_scripts/install_pico.sh
```

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/activate
pip install pyrealsense2
```

```bash
rs-enumerate-devices
v4l2-ctl --list-devices
```

## 2. Robot tmux launcher

```bash
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh
```

Переменные launcher’а:

| Переменная | Default | Возможные значения / назначение |
|---|---:|---|
| `TMUX_SESSION` | `brainco_robot` | Имя tmux-сессии. |
| `CAMERA_MODE` | `realsense-usb` | `two-realsense`, `realsense-usb`, `head-realsense`, `none`. |
| `EGO_VIEW_DEVICE_ID` | `243422071979` | Серийный номер ego RealSense. |
| `HEAD_DEVICE_ID` | `135122071874` | Серийный номер head RealSense. |
| `USB_HEAD_DEVICE_ID` | `/dev/video6` | Индекс или путь USB head camera. |
| `CAMERA_FPS` | `15` | Частота ego camera и базовая частота camera server. |
| `REALSENSE_WIDTH`, `REALSENSE_HEIGHT` | `1280`, `720` | RGB-разрешение ego RealSense. |
| `HEAD_CAMERA_WIDTH`, `HEAD_CAMERA_HEIGHT` | `1600`, `896` | Отдельное RGB-разрешение head USB/RealSense. |
| `HEAD_CAMERA_FPS` | `15` | Отдельная частота head USB/RealSense. |
| `HEAD_CAMERA_QUALITY` | `80` | JPEG quality head-потока, от `1` до `100`. |
| `HEAD_CAMERA_FOURCC` | `MJPG` | Формат захвата USB head camera. Для RealSense игнорируется. |
| `REALSENSE_DEPTH` | `0` | `0` — depth sensor не запускается и depth не отправляется; `1` — включить. |
| `HEAD_REALSENSE_DEPTH_WIDTH`, `HEAD_REALSENSE_DEPTH_HEIGHT` | `640`, `480` | Depth-разрешение для `head-realsense`. |
| `PICO_INTERFACE` | `wlxfc23cd997021` | Сетевой интерфейс для BrainCo DDS. |
| `PICO_PORT` | `5556` | Порт Pico manager. |
| `TELEMETRY_PORT` | `5560` | Порт G1 telemetry publisher. |
| `TELEMETRY_HZ` | `50` | Целевая частота telemetry. |
| `BRAINCO_CONTAINER` | `g1-brainco-hand-server` | Имя Docker container. |
| `BRAINCO_MAX_ATTEMPTS` | `10` | Число проверок BrainCo state перед ошибкой. |
| `BRAINCO_DDS_INTERFACE` | `PICO_INTERFACE` | DDS-интерфейс для BrainCo state check. |
| `TELEOP_VENV`, `CAMERA_VENV`, `DEPLOY_DIR` | пути Unitree | Пути окружений и deploy. |

Примеры:

```bash
# ego RealSense + USB head camera
CAMERA_MODE=realsense-usb \
EGO_VIEW_DEVICE_ID=243422071979 \
USB_HEAD_DEVICE_ID=/dev/video6 \
HEAD_CAMERA_WIDTH=1600 \
HEAD_CAMERA_HEIGHT=896 \
HEAD_CAMERA_FPS=15 \
HEAD_CAMERA_QUALITY=90 \
HEAD_CAMERA_FOURCC=MJPG \
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace

# camera server запускается отдельно либо на другой машине
CAMERA_MODE=none \
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace

# Только head RealSense: RGB 1280x960, без depth
CAMERA_MODE=head-realsense \
HEAD_DEVICE_ID=135122071874 \
HEAD_CAMERA_WIDTH=1280 \
HEAD_CAMERA_HEIGHT=960 \
HEAD_CAMERA_FPS=15 \
HEAD_CAMERA_QUALITY=90 \
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace
```

tmux-окна robot launcher: `brainco`, `camera` (кроме `CAMERA_MODE=none`),
`pico`, `telemetry`, `deploy`.

## 3. Camera server

### Two RealSense

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --ego-view-device-id 243422071979 \
  --head-camera realsense \
  --head-device-id 135122071874 \
  --realsense-width 1280 \
  --realsense-height 720 \
  --head-camera-width 1280 \
  --head-camera-height 720 \
  --head-camera-fps 15 \
  --head-camera-quality 80 \
  --no-realsense-depth \
  --fps 15 \
  --port 5555
```

Основные аргументы: `--ego-view-camera`, `--ego-view-device-id`,
`--head-camera`, `--head-device-id`, `--realsense-width`,
`--realsense-height`, `--head-camera-width`, `--head-camera-height`,
`--head-camera-fps`, `--head-camera-quality`, `--head-camera-fourcc`,
`--realsense-depth` / `--no-realsense-depth`, `--realsense-depth-width`,
`--realsense-depth-height`, `--fps`, `--port`.

В default-команде используется `--no-realsense-depth`: depth sensor не
запускается, depth-кадры не кодируются и не отправляются на host.

### RealSense + USB head camera

```bash
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --ego-view-device-id 243422071979 \
  --head-camera usb \
  --head-device-id /dev/video6 \
  --realsense-width 1280 \
  --realsense-height 720 \
  --head-camera-width 1280 \
  --head-camera-height 960 \
  --head-camera-fps 15 \
  --head-camera-quality 90 \
  --head-camera-fourcc MJPG \
  --no-realsense-depth \
  --fps 15 \
  --port 5555
```

### Только head RealSense

```bash
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera None \
  --head-camera realsense \
  --head-device-id 135122071874 \
  --head-camera-width 1280 \
  --head-camera-height 960 \
  --head-camera-fps 15 \
  --head-camera-quality 90 \
  --no-realsense-depth \
  --fps 15 \
  --port 5555
```

`None` здесь чувствителен к регистру. Для opt-in depth используйте
`--realsense-depth` и задайте поддерживаемые размеры через
`--realsense-depth-width` / `--realsense-depth-height`.

Если `--head-camera usb`, настройки depth относятся только к другим RealSense
камерам: USB head никогда не создаёт и не отправляет `head_depth`. Перед выбором
разрешения/FPS проверьте поддерживаемые сочетания командой
`v4l2-ctl --device /dev/video6 --list-formats-ext`.

## 4. BrainCo service

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/start_brainco_and_check.sh
```

Скрипт перезапускает `g1-brainco-hand-server`, если container уже запущен;
иначе запускает его. Затем проверяет состояния обеих рук и при ошибке снова
перезапускает container. Максимум — 10 проверок.

Аргументы: `--container`, `--attempts`, `--interface`, `--restart-wait`,
`--python`.

Примеры:

```bash
# Другой DDS-интерфейс
bash gear_sonic/scripts/start_brainco_and_check.sh \
  --interface wlxfc23cd997021

# Другой container и пять попыток
BRAINCO_CONTAINER=my-brainco-service \
bash gear_sonic/scripts/start_brainco_and_check.sh \
  --container my-brainco-service \
  --attempts 5

docker stop g1-brainco-hand-server
```

## 5. Pico manager

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.scripts.pico_manager_brainco_dexterous \
  --manager \
  --port 5556 \
  --target_fps 50 \
  --brainco_network_interface wlxfc23cd997021
```

Основные аргументы: `--manager`, `--port`, `--target_fps`,
`--input-source {xrt,isaac-teleop}`, `--brainco_dds_domain`,
`--brainco_network_interface`, `--brainco_trigger_threshold`,
`--brainco_trigger_range`, `--brainco_excluded_fingers`,
`--disable_brainco_hand`, `--zmq_feedback_host`, `--zmq_feedback_port`.

Пример порога trigger:

```bash
--brainco_trigger_threshold 0.5
```

## 6. G1 upper-body telemetry publisher

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.g1_upper_body_telemetry.robot_publisher \
  --publish-hz 50 \
  --port 5560
```

Аргументы: `--publish-hz`, `--bind-host`, `--port`, `--dds-domain-id`,
`--network-interface`, `--state-topic`, `--command-topic`, `--stale-seconds`.

## 7. Gear Sonic deploy

```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd /home/unitree/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
```

Основные аргументы: `--input-type zmq_manager`, режим `real` или `sim`.

## 8. Video streaming server

### RealSense

```bash
cd /home/unitree/teleop-ws/XRoboToolkit-Orin-Video-Sender
REALSENSE_FORMAT=YUYV \
REALSENSE_WIDTH=640 \
REALSENSE_HEIGHT=480 \
REALSENSE_FPS=30 \
./server_realsense.sh 192.168.50.132
```

Аргументы задаются переменными: `REALSENSE_DEVICE`, `REALSENSE_FORMAT`,
`REALSENSE_WIDTH`, `REALSENSE_HEIGHT`, `REALSENSE_FPS`; последний аргумент —
IP host-компьютера.

---

# Host-компьютер

## 1. Подготовка

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection
bash install_scripts/install_data_collection.sh --replace
```

## 2. Host tmux launcher

```bash
ROBOT_HOST=192.168.50.132 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace
```

Переменные launcher’а:

| Переменная | Default | Возможные значения / назначение |
|---|---:|---|
| `TMUX_SESSION` | `brainco_host` | Имя tmux-сессии. |
| `ROBOT_HOST` | `192.168.50.132` | Общий адрес робота. |
| `CAMERA_HOST`, `SONIC_HOST`, `STATE_HOST`, `TELEMETRY_HOST` | `ROBOT_HOST` | Адреса отдельных ZMQ источников. |
| `CAMERA_PORT`, `SONIC_PORT`, `STATE_PORT`, `TELEMETRY_PORT` | `5555`, `5556`, `5557`, `5560` | Порты источников. |
| `PICO_INTERFACE` | `wlp128s20f3` | DDS-интерфейс BrainCo на host. |
| `ENABLE_CAMERA_VIEWER` | `1` | `1` — создать окно viewer, `0` — не создавать. |
| `SHOW_MOTOR_STATES` | `0` | `0` — скрыть motor-state/tau панель; `1` — включить её и соответствующие subscribers. |
| `CAMERA_STREAMS` | `head ego_view` | RGB-потоки робота для viewer; external view добавляется автоматически. |
| `HEAD_CAMERA_WIDTH`, `HEAD_CAMERA_HEIGHT` | `1600`, `896` | Ожидаемый exporter размер robot head stream; должен совпадать с robot launcher. |
| `EXPORTER_EXTRA_ARGS` | пусто | Дополнительные аргументы exporter. |
| `EXTERNAL_VIEW_CAMERA_DEVICE` | `/dev/video4` | Путь к локальной камере host; пустая строка отключает её. |
| `EXTERNAL_VIEW_CAMERA_WIDTH`, `EXTERNAL_VIEW_CAMERA_HEIGHT`, `EXTERNAL_VIEW_CAMERA_FPS` | `1280`, `960`, `15` | Запрашиваемый V4L2-профиль локальной камеры. |
| `EXTERNAL_VIEW_CAMERA_FOURCC` | `MJPG` | V4L2 FourCC внешней камеры. |
| `EXTERNAL_VIEW_CAMERA_HOST`, `EXTERNAL_VIEW_CAMERA_PORT` | `localhost`, `5582` | ZMQ-поток standalone external-view-camera для exporter и viewer. |
| `DATA_COLLECTION_VENV` | `.venv_data_collection` | Путь к окружению exporter. |

Примеры:

```bash
# Только ego_view, без viewer
ROBOT_HOST=192.168.50.132 \
ENABLE_CAMERA_VIEWER=0 \
EXPORTER_EXTRA_ARGS="--ignore-head --record-raw-depth" \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace

# Источники находятся на разных машинах
CAMERA_HOST=192.168.50.132 \
SONIC_HOST=192.168.50.132 \
STATE_HOST=192.168.50.132 \
TELEMETRY_HOST=192.168.50.132 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace

# Локальная external-view-camera на host в 1280x960
EXTERNAL_VIEW_CAMERA_DEVICE=/dev/video4 \
EXTERNAL_VIEW_CAMERA_WIDTH=1280 \
EXTERNAL_VIEW_CAMERA_HEIGHT=960 \
EXTERNAL_VIEW_CAMERA_FPS=15 \
EXTERNAL_VIEW_CAMERA_FOURCC=MJPG \
SHOW_MOTOR_STATES=0 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace
```

tmux-окна host launcher: `exporter`, `external_camera` (если задано
`EXTERNAL_VIEW_CAMERA_DEVICE`) и `viewer` (если `ENABLE_CAMERA_VIEWER=1`).

### Standalone external-view-camera

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python gear_sonic/scripts/run_external_view_camera.py \
  --device /dev/video4 \
  --width 1280 \
  --height 960 \
  --fps 15 \
  --fourcc MJPG \
  --port 5582
```

Этот процесс единственный открывает `/dev/video4`. Exporter и viewer
независимо подписываются на его ZMQ-поток.

## 3. BrainCo data exporter

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python -m gear_sonic.scripts.brainco_data_exporter \
  --camera-host 192.168.50.132 \
  --camera-port 5555 \
  --sonic-zmq-host 192.168.50.132 \
  --sonic-zmq-port 5556 \
  --state-zmq-host 192.168.50.132 \
  --state-zmq-port 5557 \
  --g1-telemetry-zmq-host 192.168.50.132 \
  --g1-telemetry-zmq-port 5560 \
  --g1-telemetry-expected-hz 50 \
  --record-raw-telemetry \
  --depth-camera-width 1280 \
  --depth-camera-height 720 \
  --webcam-width 1600 \
  --webcam-height 896 \
  --brainco-network-interface wlp128s20f3 \
  --external-view-camera-host localhost \
  --external-view-camera-port 5582 \
  --external-view-camera-width 1280 \
  --external-view-camera-height 960
```

Перед подключением к источникам exporter запрашивает task, пол,
возраст, рост и вес. После проверки можно подтвердить данные,
изменить одно поле или ввести всё заново. Имя датасета формируется
автоматически.
Допустимы task `none`, `handshake`, `fist_bump`, `hug`; пол `f` или `m`;
возраст 1–120 лет; рост 120–230 см; вес 40–120 кг.
Для возраста, роста и веса можно ввести `n`; в metadata и имени датасета
это значение будет записано как `none`.
Пример имени:

```text
handshake-m-age28-height182-weight78-20260910-153045
```

Основные аргументы:

| Группа | Аргументы |
|---|---|
| Dataset | `--root-output-dir`, `--data-collection-frequency`; task и имя задаются интерактивно. |
| ZMQ | `--camera-host`, `--camera-port`, `--sonic-zmq-host`, `--sonic-zmq-port`, `--state-zmq-host`, `--state-zmq-port`. |
| BrainCo DDS | `--brainco-dds-domain-id`, `--brainco-network-interface`, `--brainco-message-timeout`. |
| G1 telemetry | `--g1-telemetry-zmq-host`, `--g1-telemetry-zmq-port`, `--g1-telemetry-expected-hz`, `--g1-telemetry-max-age`, `--g1-telemetry-message-timeout`. |
| Cameras | `--ignore-head`, `--ignore-ego-view`, `--record-wrist-cameras`, размеры RGB/depth потоков. |
| External camera stream | `--external-view-camera-host`, `--external-view-camera-port`, `--external-view-camera-width`, `--external-view-camera-height`, `--external-view-camera-timeout`. |
| Depth | `--record-raw-depth`, `--record-depth-video`, `--depth-video-max-meters`. |
| Raw telemetry | `--record-raw-telemetry`, `--no-record-raw-telemetry`. |
| Other | `--text-to-speech`, `--no-text-to-speech`, `--episode-status-zmq-port`. |

Одновременно `--ignore-head` и `--ignore-ego-view` использовать нельзя.

Примеры:

```bash
# Только ego_view и raw depth только этой камеры
--ignore-head --record-raw-depth

# Только head camera
--ignore-ego-view

# Не-default: явно включить сохранение уже доступных depth-потоков
--record-raw-depth --record-depth-video

# Standalone camera stream. В датасете modality называется external-view-camera.
--external-view-camera-host localhost \
--external-view-camera-port 5582 \
--external-view-camera-width 1280 \
--external-view-camera-height 960
```

## 4. G1 telemetry diagnostic receiver

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python -m gear_sonic.g1_upper_body_telemetry.host_receiver \
  --host 192.168.50.132 \
  --port 5560 \
  --timeout 10
```

Аргументы: `--host`, `--port`, `--timeout`.

Этот receiver не требуется для exporter: exporter сам подписывается на тот же
ZMQ topic и может работать одновременно с diagnostic receiver.

Проверка raw `rt/lowstate` напрямую через Unitree Python SDK запускается на
роботе:

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate

python -m gear_sonic.g1_upper_body_telemetry.inspect_upper_body_state \
  --print-hz 2
```

При запуске с другого компьютера, который видит DDS, добавьте интерфейс:

```bash
--network-interface wlp128s20f3
```

Скрипт выводит state `q`, `dq`, `ddq`, `tau_est` и command `q`, `dq`,
`tau`, `kp`, `kd` для 17 upper-body моторов. `--once` напечатает одну
пару LowState/LowCmd и завершится.

## 5. Camera viewer

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 192.168.50.132 \
  --camera-port 5555 \
  --external-view-camera-host localhost \
  --external-view-camera-port 5582 \
  --camera-streams head ego_view \
  --no-depth \
  --grid-columns 2 \
  --no-motor-states
```

Основные аргументы: `--camera-host`, `--camera-port`, `--camera-streams`,
`--external-view-camera-host`, `--external-view-camera-port`, `--grid-columns`,
`--depth-display-min-meters`, `--depth-display-max-meters`,
`--show-tau-plot` / `--no-show-tau-plot` и алиас `--no-motor-states`.

Viewer и exporter получают `external-view-camera` напрямую от
standalone publisher. Устройство `/dev/video*` открывает только
`run_external_view_camera.py`, поэтому перезапуск exporter камеру не закрывает.

Для одной ego camera:

```bash
--camera-streams ego_view --no-depth
```

## 6. Dataset paths

```text
outputs/<dataset-name>/
raw-telemetry/chunk-NNN/episode_NNNNNN.npz
meta/interaction_metadata.jsonl
```

`meta/interaction_metadata.jsonl` содержит отдельную строку для
каждого эпизода: профиль участника, `interaction_class` и нулевые
`approach_start`, `contact_active_start`, `release_start`, `idle_start` для
последующей разметки.

В Parquet и `raw-telemetry` для upper body сохраняются `q_cmd`, `kp`, `kd`,
`q_est`, `dq_est`, `tau_est`, `q_residual` и служебные
timestamp/sequence ID. Изменение применяется только к новым эпизодам.

В default-профиле папки `depth/` и `depth_head/` не создаются.

## 7. Управление эпизодами и временными метками

- `левый grip + A` — начать или остановить запись эпизода;
- `левый grip + B` — сохранить текущий эпизод как discarded;
- нажатие правого grip — поставить временную метку;
- удержание правого grip создаёт одну метку, используется фронт нажатия;
- левый grip сам по себе метку не создаёт и остаётся частью команд `A/B`;
- нажатия и отпускания левого и правого trigger сохраняются независимо;
- граница нажатия trigger задаётся в Pico manager через
  `--brainco_trigger_threshold` и по умолчанию равна `0.5`.

---

## Копирование проекта на Unitree по SSH

```bash
rsync -avh --partial --info=progress2 \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  /home/nikita/Skoltech/MWS/Teleop-Data-Collection/ \
  unitree@192.168.50.132:/home/unitree/teleop-ws/Teleop-Data-Collection/
```

---

## Troubleshooting / устранение проблем

### Робот подлагивает при телеуправлении через Pico

Захват камеры и JPEG-кодирование могут конкурировать за CPU с Pico и inference
GearSonic. На роботе найдите фактические PID процессов:

```bash
pgrep -af 'deploy|gear_sonic|composed_camera|pico_manager'
```

После старта всех tmux-окон проще всего запустить helper: он сам находит PID,
запрашивает `sudo` один раз и выставляет нужные значения.

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/set_runtime_priorities.sh
```

Helper требует ровно по одному процессу GearSonic deploy, Pico manager, camera
server и telemetry publisher. Если одновременно работают две tmux-сессии, он
ничего не меняет и выводит найденные дубликаты.

При необходимости те же значения можно применить вручную. Поднимите
CPU-приоритет фактического бинарника `g1_deploy_onnx_ref` и Pico manager, а
для camera server снизьте его. Замените заполнители на PID из команды выше:

```bash
sudo renice -n -5 -p <DEPLOY_PID>
sudo renice -n -5 -p <PICO_MANAGER_PID>
sudo renice -n 10 -p <CAMERA_SERVER_PID>
sudo renice -n 5 -p <TELEMETRY_PID>
```

Например, если у `g1_deploy_onnx_ref` PID `17612`, у Pico manager — `16999`,
у camera server — `17012`, а у telemetry publisher — `17010`:

```bash
sudo renice -n -5 -p 17612
sudo renice -n -5 -p 16999
sudo renice -n 10 -p 17012
sudo renice -n 5 -p 17010
```

Проверить применённые приоритеты:

```bash
ps -o pid,ni,cls,cmd -p <DEPLOY_PID>,<PICO_MANAGER_PID>,<CAMERA_SERVER_PID>,<TELEMETRY_PID>
```

`nice` меняет только планирование CPU: TensorRT/GPU kernels специального
приоритета не получают. После перезапуска PID и приоритеты меняются, поэтому
команды нужно повторить после создания новой tmux-сессии. Не используйте
realtime scheduling через `chrt` без отдельной проверки: он может вытеснить
DDS/Pico процессы и ухудшить задержку управления.

### Проверка BrainCo DDS не проходит

Если `start_brainco_and_check.sh` не проходит DDS-проверку, остановите его
через `Ctrl+C` (либо дождитесь завершения), затем один раз вручную остановите
container:

```bash
docker stop g1-brainco-hand-server
```

После этого заново запустите check script. Он сам поднимет container из
чистого остановленного состояния:

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/start_brainco_and_check.sh \
  --interface wlxfc23cd997021
```

Вместо `wlxfc23cd997021` укажите фактический интерфейс Pico/DDS, если он
отличается.
