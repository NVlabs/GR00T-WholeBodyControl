# Joint Tau

Инструкция по телеоперации с RGB RealSense, RGB USB head camera, lossless
depth PNG16 и grip timestamps: [TELEOP_DEPTH_USB.md](TELEOP_DEPTH_USB.md).

Утилиты для просмотра и анализа моментов (`tau_est`) суставов Unitree G1 и моторов BrainCo. Скрипты работают только на чтение и не отправляют команды роботу.

## Зависимости

```bash
pip install matplotlib numpy pyarrow
```

Для live-записи также нужен `unitree_sdk2py` и настроенный CycloneDDS.

## Построение графика

Из записанного Parquet-эпизода:

```bash
python extend_scripts/plot_g1_joint_tau.py \
  --parquet-file outputs/<dataset>/data/chunk-000/episode_000000.parquet \
  --output joint_tau.png
```

Из live DDS (индексы суставов обязательны):

```bash
python extend_scripts/plot_g1_joint_tau.py <network-interface> \
  --joint-indices 12 15 18 22 25 --duration 20 \
  --output joint_tau.png
```

Результат сохраняется как график и CSV с тем же именем. Отключить CSV можно флагом `--no-csv`.

При наличии данных кистей дополнительно создаются `<name>_hands.png` и `<name>_hands.csv`. На графике кистей левая и правая руки расположены в двух столбцах. В live-режиме данные читаются из `rt/brainco/left/state` и `rt/brainco/right/state`; в Parquet ожидаются `observation.left_hand.tau_est` и `observation.right_hand.tau_est`. Отключить обработку кистей можно флагом `--no-hands`.

Для поиска внешних воздействий также создаются `<name>_normalized.png` и `<name>_hands_normalized.png`. По умолчанию используется rolling median/MAD baseline за 1 секунду, сглаживание residual за 0.1 секунды и красное выделение `|score| > 3`. Настройки: `--baseline-window`, `--smoothing-window`, `--event-threshold`; отключение: `--no-normalized`.

## Средние значения по эпизодам

```bash
python extend_scripts/print_episode_joint_means.py \
  outputs/<dataset>/data/chunk-000
```

## Обработка всего датасета

```bash
./extend_scripts/run_joint_tau_analysis.sh <dataset-directory> <output-directory>
```

Для каждого эпизода создаётся отдельная папка с raw и normalized графиками upper body, а также CSV. Если в Parquet записан hand tau, рядом появятся raw и normalized графики кистей и отдельный CSV.

Все параметры доступны через `python extend_scripts/<script>.py --help`.
