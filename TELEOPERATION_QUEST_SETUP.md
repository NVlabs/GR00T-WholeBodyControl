# Meta Quest 3 — Teleoperación con Sonic

Guía completa para usar **Meta Quest 3** como controlador de teleoperación con GEAR-SONIC (G1 robot).

---

## Requisitos

- **Meta Quest 3** con controladores
- **PC** con **Ubuntu 22.04** (probado aquí)
- Acceso a Internet (para instalar dependencias)
- **ADB** (Android Debug Bridge)
- **Conexión**: USB o WiFi (misma red que el Quest)

---

## 0. Dependencias de sistema (Ubuntu 22.04)

Instala las herramientas básicas que usa este repo y el script `install_pico.sh`:

```bash
sudo apt update
sudo apt install -y \
    git git-lfs \
    build-essential \
    curl \
    python3 python3-venv python3-pip \
    libgl1 libegl1 libx11-6
```

- `git` / `git-lfs`: para clonar este repo (usa ficheros grandes, vídeos, etc.).
- `build-essential`: compilar librerías nativas (XRoboToolkit, etc.).
- `curl`: necesario para que `install_pico.sh` pueda instalar `uv`.
- `python3*`: utilidades básicas de Python (aunque `uv` instalará su propio Python 3.10 gestionado).
- `libgl*` y `libx11-6`: librerías gráficas mínimas para PyVista/VTK (visualización 3D).

---

## 1. Instalar ADB

```bash
sudo apt update
sudo apt install android-tools-adb android-tools-fastboot
```

Verificar:

```bash
adb version
adb devices
```

### Permisos USB (si aparece "no permissions")

```bash
sudo bash -c 'cat > /etc/udev/rules.d/51-android.rules' << 'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

Desconecta y vuelve a conectar el Quest. Acepta el diálogo de depuración USB en el visor.

---

## 2. Configurar el Quest 3

1. **Modo desarrollador**: Actívalo desde la app Meta Quest en el móvil (Ajustes → Modo desarrollador).
2. **En el Quest**: Ajustes → Sistema → **Developer** → activa **USB Debugging**.
3. **APK de teleop**: El paquete `meta_quest_teleop` instala automáticamente el APK al conectarte. Acepta la instalación cuando aparezca.

---

## 3. Clonar e instalar el proyecto

```bash
git clone https://github.com/josue99999/TELEOPERATION_SONIC.git
cd TELEOPERATION_SONIC
git lfs pull
```

### Entorno de teleoperación (incluye meta_quest_teleop)

```bash
bash install_scripts/install_pico.sh
```

Esto crea `.venv_teleop` con `gear_sonic[teleop]`, que ya incluye `meta_quest_teleop`.

Activar:

```bash
source .venv_teleop/bin/activate
```

---

## 4. Probar la conexión

```bash
# Con Quest conectado por USB
python -m gear_sonic.utils.teleop.readers.quest_reader --test-adb
```

Deberías ver `ADB connection OK` y el modelo del dispositivo.

---

## 5. Validar tracking (visualización 3D)

Antes de usar Sonic, comprueba que el tracking del Quest funciona:

```bash
# USB
python -m gear_sonic.utils.teleop.readers.validate_quest_raw

# WiFi (sustituye por la IP de tu Quest)
python -m gear_sonic.utils.teleop.readers.validate_quest_raw --ip-address 192.168.x.x
```

Mueve los mandos y verifica que la visualización sigue. Ejes: **X=adelante (rojo)**, **Y=izquierda (verde)**, **Z=arriba (azul)**.

### Modo calibrado (G1 como referencia)

```bash
python -m gear_sonic.utils.teleop.readers.validate_quest_raw --calibrated
```

1. Pon el G1 en su **pose por defecto**.
2. Adopta la **misma pose** con los controladores.
3. Pulsa el **trigger derecho** para calibrar.
4. A partir de ahí, la salida es: `pose_G1_default + delta_movimiento`.

---

## 6. Usar Quest con Sonic

```bash
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --reader quest
```

Por WiFi:

```bash
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --reader quest \
    --quest-ip-address 192.168.x.x
```

**Calibración**: Igual que en `validate_quest_raw`: adopta la pose por defecto del G1 y pulsa **trigger derecho**.

---

## 7. Ejecutar todo completo (sim + deploy + Quest)

Abre **3 terminales** y ejecuta en este orden:

### Terminal 1 — Simulación MuJoCo

```bash
cd TELEOPERATION_SONIC   # o la ruta donde clonaste el repo
source .venv_sim/bin/activate   # o .venv_teleop si solo usaste install_pico.sh
python gear_sonic/scripts/run_sim_loop.py
```

### Terminal 2 — Deploy (C++ inference)

```bash
cd TELEOPERATION_SONIC/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh sim --input-type zmq_manager
```

Espera hasta ver **"Init done"**.

### Terminal 3 — Manager con Quest

```bash
cd TELEOPERATION_SONIC
source .venv_teleop/bin/activate

# USB (recomendado la primera vez)
python gear_sonic/scripts/pico_manager_thread_server.py --manager --reader quest

# Con visualización VR 3PT
python gear_sonic/scripts/pico_manager_thread_server.py --manager --reader quest --vis_vr3pt

# WiFi (sustituye por la IP de tu Quest)
python gear_sonic/scripts/pico_manager_thread_server.py --manager --reader quest --quest-ip-address 192.168.x.x
```

**Orden**: 1 → 2 → 3. Calibra con el **trigger derecho** cuando el manager esté corriendo.

---

## Resumen de comandos

| Acción | Comando |
|--------|---------|
| Probar ADB | `python -m gear_sonic.utils.teleop.readers.quest_reader --test-adb` |
| Validar tracking (raw) | `python -m gear_sonic.utils.teleop.readers.validate_quest_raw` |
| Validar tracking (calibrado) | `python -m gear_sonic.utils.teleop.readers.validate_quest_raw --calibrated` |
| Sonic con Quest | `python gear_sonic/scripts/pico_manager_thread_server.py --manager --reader quest` |

---

## Solución de problemas

- **"No devices found"**: Revisa USB, modo desarrollador y depuración USB. Prueba `adb kill-server && adb start-server`.
- **"meta_quest_teleop no instalado"**: Ejecuta `bash install_scripts/install_pico.sh` y activa `.venv_teleop`.
- **Las manos van en dirección equivocada**: El reader corrige los ejes automáticamente. Si algo falla, usa `validate_quest_raw --calibrated` para comprobar.
