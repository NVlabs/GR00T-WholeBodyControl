# Телеоперация BrainCo с RealSense и head camera

Инструкция описывает две конфигурации камер:

- RealSense `ego_view` + обычная USB-камера `head`;
- две RealSense: `ego_view` и `head`, каждая с RGB и depth.

Во второй конфигурации exporter может дополнительно сохранять оба depth-потока
как lossless PNG16 в миллиметрах и как grayscale MP4 preview. Значение depth `0`
означает невалидный пиксель.

Все команды проекта ниже запускаются из корня `Teleop-Data-Collection`.

## 1. Подготовка

Если окружения ещё не созданы:

```bash
bash install_scripts/install_camera_server.sh
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh
```

RealSense SDK устанавливается в camera-окружение отдельно:

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/activate
pip install pyrealsense2
```

Перед запуском должны работать:

- G1 deploy/ZMQ state publisher на порту `5557`;
- G1 upper-body telemetry publisher на роботе на порту `5560`;
- `brainco_hand_service` и DDS-топики обеих кистей;
- Pico/CloudXR либо другой выбранный input source.

Проверьте серийные номера RealSense и индекс USB-камеры:

```bash
rs-enumerate-devices
v4l2-ctl --list-devices
```

RealSense создаёт несколько `/dev/video*`. Индекс USB head camera выбирайте по
выводу `v4l2-ctl`, не предполагая, что это всегда `0` или `1`.

## 2. Camera server

В отдельном терминале активируйте окружение и перейдите в репозиторий:

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection
```

### Вариант A: RealSense + USB head camera

```bash
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --ego-view-device-id 243422071979 \
  --head-camera usb \
  --head-device-id /dev/video6 \
  --realsense-width 640 \
  --realsense-height 480 \
  --head-camera-width 1600 \
  --head-camera-height 896 \
  --head-camera-fps 15 \
  --head-camera-quality 90 \
  --head-camera-fourcc MJPG \
  --realsense-depth-width 640 \
  --realsense-depth-height 480 \
  --fps 30 \
  --port 5555
```

Замените серийный номер и `/dev/video6` своими значениями. Вместо пути можно
передать числовой индекс. Сервер
публикует `ego_view`, `ego_view_depth` и `head`.

Для USB head настройки `--realsense-depth-*` автоматически не применяются:
USB-камера публикует только `head`, без `head_depth`. Поддерживаемые USB-режимы
проверяются через `v4l2-ctl --device /dev/video6 --list-formats-ext`; для высоких
разрешений обычно требуется `--head-camera-fourcc MJPG`.

В текущем exporter флаги `--record-depth-video` и `--record-raw-depth` являются
общими для ego и head. Поэтому для пары RealSense ego + USB head оставляйте их
выключенными, если exporter ожидает оба depth-ключа.

### Вариант B: две RealSense

```bash
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
  --realsense-depth-width 640 \
  --realsense-depth-height 480 \
  --fps 15 \
  --port 5555
```

Для нескольких RealSense укажите разные серийные номера обеих камер. Сервер
публикует четыре ключа:

```text
ego_view        RGB ego-view RealSense
ego_view_depth  depth ego-view RealSense
head            RGB head RealSense
head_depth      depth head RealSense
```

Depth выравнивается в координаты соответствующего RGB-потока и передаётся как
lossless `uint16_mm_png`.

### Просмотр потоков

На машине с дисплеем:

```bash
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 192.168.50.132 \
  --camera-port 5555 \
  --camera-streams head head_depth ego_view ego_view_depth \
  --grid-columns 2 \
  --depth-display-min-meters 0.2 \
  --depth-display-max-meters 5.0
```

Для варианта с USB удалите `head_depth` из `--camera-streams`. Параметры
визуализации не влияют на PNG16, записываемые exporter.

## 3. BrainCo hand server

```bash
docker start g1-brainco-hand-server
```

Остановка сервиса:

```bash
docker stop g1-brainco-hand-server
```

## 4. Pico manager

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.scripts.pico_manager_brainco_dexterous \
  --manager \
  --port 5556 \
  --target_fps 50 \
  --brainco_network_interface wlxfc23cd997021
```

Старый путь `gear_sonic/scripts/pico_manager_thread_server_brainco_dexterous_adaptive_control.py`
оставлен как совместимый wrapper и принимает те же аргументы.

## 5. G1 upper-body telemetry

DDS `rt/lowstate` доступен локально на роботе, поэтому publisher запускается на
G1. Он читает `LowState` и `LowCmd` и с заданной частотой отправляет на host
`q/dq/ddq/tau` для измеренного и командного состояния, PD-момент и residual'ы.

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate

python -m gear_sonic.g1_upper_body_telemetry.robot_publisher \
  --publish-hz 50 \
  --port 5560
```

Если на робота скопирована только папка `g1_upper_body_telemetry`:

```bash
cd /home/unitree/nikita
python -m g1_upper_body_telemetry.robot_publisher --publish-hz 50 --port 5560
```

Проверка потока на host-компьютере:

```bash
python -m gear_sonic.g1_upper_body_telemetry.host_receiver \
  --host 192.168.50.132 \
  --port 5560
```

В новые датасеты из G1 telemetry сохраняются `q_cmd`, `kp`, `kd`,
`q_est`, `dq_est`, `tau_est` и `q_residual`, а также timestamp,
sequence ID и age исходных DDS-сообщений. Residual'ы имеют знак
`estimated - commanded`. Тот же набор числовых полей записывается в
`raw-telemetry/*.npz`; каждый пакет по-прежнему сохраняется со своими
временными метками.

## 6. Gear Sonic deploy

```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd /home/unitree/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
```

## 7. Video streaming server

### RealSense

```bash
cd /home/unitree/teleop-ws/XRoboToolkit-Orin-Video-Sender
REALSENSE_FORMAT=YUYV \
REALSENSE_WIDTH=640 \
REALSENSE_HEIGHT=480 \
REALSENSE_FPS=30 \
./server_realsense.sh 192.168.50.132
```

### USB webcam

```bash
cd /home/unitree/teleop-ws/XRoboToolkit-Orin-Video-Sender
REALSENSE_DEVICE=/dev/video6 \
REALSENSE_FORMAT=MJPG \
REALSENSE_WIDTH=1280 \
REALSENSE_HEIGHT=720 \
REALSENSE_FPS=30 \
./server_realsense.sh 192.168.50.132
```

## 8. Data exporter

В отдельном терминале:

```bash
source .venv_data_collection/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection
```

### RealSense + USB head camera

```bash
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
  --brainco-network-interface wlp128s20f3 \
  --depth-camera-width 640 \
  --depth-camera-height 480 \
  --webcam-width 1600 \
  --webcam-height 896
```

### Две RealSense с записью depth

Размеры обязаны совпадать с параметрами camera server. В приведённой выше
конфигурации RGB имеет размер `1280x720`, а depth — `640x480`.

```bash
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
  --brainco-network-interface wlp128s20f3 \
  --depth-camera-width 1280 \
  --depth-camera-height 720 \
  --depth-stream-width 640 \
  --depth-stream-height 480 \
  --webcam-width 1280 \
  --webcam-height 720 \
  --head-depth-stream-width 640 \
  --head-depth-stream-height 480 \
  --record-depth-video \
  --record-raw-depth
```

Старый путь `gear_sonic/scripts/run_data_exporter_brainco_extented.py` сохранён
как совместимый wrapper. Для каждой новой схемы или сессии используйте новый
`dataset-name`: старый датасет не обновляет схему автоматически.

`--depth-video-max-meters` влияет только на MP4 preview и не изменяет PNG16.

Можно исключить один из основных camera-потоков из датасета:

```bash
# Только head camera
--ignore-ego-view

# Только ego-view camera
--ignore-head
```

Одновременное использование `--ignore-ego-view --ignore-head` запрещено: хотя
бы одна из этих двух камер должна оставаться в датасете. Флаги исключают как RGB,
так и соответствующие depth-данные. Поэтому, например,
`--ignore-head --record-raw-depth` сохраняет только `ego_view_depth` и не требует
наличия `head_depth`.

По умолчанию каждый пакет telemetry, пришедший во время эпизода, сохраняется в
`raw-telemetry/chunk-NNN/episode_NNNNNN.npz`. В NPZ находятся исходные команды и
измерения, PD-момент, residual'ы, timestamp'ы, sequence id, возраст DDS-сообщений,
имена и индексы суставов. Отключить эту запись можно флагом
`--no-record-raw-telemetry`. Один последний свежий telemetry-сэмпл также
синхронизируется с каждым кадром основного датасета.

## 9. Управление эпизодами и временными метками

- `левый grip + A` — начать или остановить запись эпизода;
- `левый grip + B` — сохранить текущий эпизод как discarded;
- нажатие правого grip — поставить временную метку;
- удержание правого grip создаёт одну метку, используется фронт нажатия;
- левый grip сам по себе метку не создаёт и остаётся частью команд `A/B`;
- нажатия и отпускания левого и правого trigger сохраняются независимо;
- граница нажатия trigger задаётся в Pico manager через
  `--brainco_trigger_threshold` и по умолчанию равна `0.5`.

## 10. Структура после рефакторинга

Pico manager находится в `gear_sonic/scripts/pico_manager_brainco_dexterous/`:

- `brainco.py` и `g1_state.py` — DDS и управление кистями/состоянием G1;
- `controller_input.py` — Pico/XRT и Isaac Teleop input;
- `pose_processing.py`, `tracking.py`, `pose_streamer.py` — обработка и передача поз;
- `planner.py`, `manager.py` — planner loop и state machine;
- `cli.py` — аргументы командной строки.

Exporter находится в `gear_sonic/scripts/brainco_data_exporter/`:

- `config.py` и `constants.py` — конфигурация и схема сигналов;
- `dds.py` — подписка на BrainCo;
- `schema.py` — 41-DOF и camera schema;
- `exporter.py` — запись эпизодов и depth;
- `collector.py` — сбор и синхронизация потоков;
- `main.py`, `cli.py` — сборка приложения и CLI.

Отдельный робот-host telemetry transport находится в
`gear_sonic/g1_upper_body_telemetry/`.

## 11. Копирование проекта на Unitree по SSH

```bash
rsync -avh --partial --info=progress2 \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  /home/nikita/Skoltech/MWS/Teleop-Data-Collection/ \
  unitree@192.168.50.132:/home/unitree/teleop-ws/Teleop-Data-Collection/
```
