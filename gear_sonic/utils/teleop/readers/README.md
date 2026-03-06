# Quest 3 reader

Reader de body pose para Quest 3, compatible con el formato de `PicoReader` para teleop.

## Probar la conexión ADB (Quest 3 por USB)

Con el Quest 3 conectado y depuración USB activada:

```bash
# Desde la raíz del repo, usar el venv del proyecto
.venv_sim/bin/python -m gear_sonic.utils.teleop.readers.quest_reader --test-adb
```

Si hay dispositivo: verás "ADB connection OK" y el modelo. Si no: "No devices found".

## Probar el reader con datos sintéticos (sin Quest)

Para comprobar que el reader y el formato de `sample` funcionan:

```bash
.venv_sim/bin/python -m gear_sonic.utils.teleop.readers.quest_reader --synthetic --duration 5
```

Deberías ver algo como: `Last sample: body_poses_np shape (24, 7), fps=~50`.

## Probar Meta Quest (meta_quest_teleop)

Si tienes `meta_quest_teleop` instalado y el APK en el Quest, prueba que llegan datos de los controladores:

```bash
# Test rápido (solo imprime poses)
.venv_sim/bin/python gear_sonic/utils/teleop/readers/test_quest_meta_quest.py

# WiFi
.venv_sim/bin/python gear_sonic/utils/teleop/readers/test_quest_meta_quest.py --ip-address 192.168.x.x
```

Deberías ver `L:OK R:OK` y posiciones que cambian al mover los mandos.

## Validar datos crudos del Quest (visualización 3D)

Antes de depurar calibración o mapeo al robot, valida que el tracking del Quest sigue correctamente:

```bash
# USB — visualiza poses en 3D sin calibración
.venv_teleop/bin/python -m gear_sonic.utils.teleop.readers.validate_quest_raw

# WiFi
.venv_teleop/bin/python -m gear_sonic.utils.teleop.readers.validate_quest_raw --ip-address 192.168.x.x

# Sin mesh del G1 (solo marcadores VR)
.venv_teleop/bin/python -m gear_sonic.utils.teleop.readers.validate_quest_raw --no-g1
```

Mueve los mandos y verifica que la visualización sigue. Si sigue bien → el problema está en calibración/mapeo. Si no sigue → problema en meta_quest_teleop o Quest.

### Modo calibrado (G1 como referencia)

Además del modo raw, `validate_quest_raw` tiene un modo **calibrado** donde la referencia es la pose por defecto del G1 (FK):

```bash
# Modo calibrado (G1 FK como referencia)
.venv_teleop/bin/python -m gear_sonic.utils.teleop.readers.validate_quest_raw --calibrated

# Opcional: WiFi + calibrado
.venv_teleop/bin/python -m gear_sonic.utils.teleop.readers.validate_quest_raw --calibrated --ip-address 192.168.x.x
```

Flujo recomendado:

- Pon el robot G1 en su **pose por defecto** (la que usa Sonic al iniciar).
- Adopta **la misma pose** con tu cuerpo/controladores.
- Pulsa el **trigger derecho** para calibrar.
- A partir de ahí, la salida es: `pose_G1_default + delta_movimiento_Quest`.

Detalles importantes:

- Sistema de ejes en la visualización:
  - **X (rojo)**: adelante
  - **Y (verde)**: izquierda
  - **Z (azul)**: arriba
- El reader corrige los ejes que vienen del APK de `meta_quest_teleop` para que:
  - Adelante/atrás se vea en **X (rojo)**.
  - Izquierda/derecha se vea en **Y (verde)**.
  - Arriba/abajo se vea en **Z (azul)**.

## QuestReader con VR 3-point (formato SONIC)

El `QuestReader` con `source="meta_quest"` produce datos en el formato esperado por VR_3PT:

```bash
# Conectar Quest por USB, pulsar trigger derecho para calibrar
.venv_sim/bin/python -m gear_sonic.utils.teleop.readers.quest_reader --source meta_quest --duration 30

# WiFi
.venv_sim/bin/python -m gear_sonic.utils.teleop.readers.quest_reader --source meta_quest --ip-address 192.168.x.x --duration 30
```

El sample incluye `vr_3pt_pose` (3, 7): L-Wrist, R-Wrist, Head (por defecto). El `PlannerLoop` usa `vr_3pt_pose` directamente cuando está presente.

### Uso dentro de Sonic (pico_manager_thread_server)

Para usar Quest directamente con Sonic (misma calibración y ejes que en `validate_quest_raw`):

```bash
# Manager de Sonic usando Quest en lugar de Pico
.venv_sim/bin/python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --reader quest \
    --quest-ip-address 192.168.x.x   # omitir para USB
```

Notas:

- `--reader quest` crea un `QuestReader(source="meta_quest")` y le pasa la pose por defecto del G1 (FK) como referencia de calibración.
- La calibración en Sonic se hace igual: adopta la pose por defecto y pulsa **trigger derecho**.
- El `vr_3pt_pose` que Sonic envía al planner ya incluye:
  - La referencia de G1 FK (pose por defecto de muñecas).
  - La misma corrección de ejes que se ve en `validate_quest_raw`.

## Siguiente paso: conectar datos reales del Quest

Implementa `read_quest_body_via_adb()` en `quest_reader.py` para leer la pose desde tu app/socket/archivo del Quest y devolver un array `(N, 7)` por joint. Luego ejecuta con `--source adb` (y con el reader integrado en `pico_manager_thread_server.py` cuando lo enlaces).
