# G1 JetPack 6 Flashing Guide

## 1. Download Image

1. **Download the required files** — get both the `.tar` file and the image file from [Jetpack 6.2](https://drive.google.com/drive/folders/1ho17ectOxi7FbaRFdpAbP4tet8BJWjbm).

## 2. Unmount Orin NX's NVMe

### Steps to Remove the NVMe SSD

1. **Remove the back handle screws**

   Use a 5 mm T-handle Allen key to unscrew the two screws located at the back of the robot near the handle.

2. **Remove the foam and plastic back cover**

   - Use the 2 mm hex tool from the Fanttik tool kit to remove the four screws holding the foam and plastic backing in place.
   - Lift the backing off to expose the internal components.

```{image} ../_static/screws.png
:width: 600px
:align: center
```

3. **Remove the NVMe screw on the Orin NX module**

   Use a Phillips screwdriver (also in the Fanttik tool kit) to remove the single screw securing the Orin NX's NVMe SSD.

```{image} ../_static/ssd.png
:width: 600px
:align: center
```

4. **Remove the SSD card**

   Carefully slide out and remove the NVMe SSD from its slot.

## 3. Flash the NVMe SSD

**Mount the NVMe SSD from the Orin NX into the NVMe SSD enclosure adapter.**
(Adapter needed when burning image from a laptop)

1. Check that the robot's SSD is unmounted. Run the following command to make sure the external SSD (where you will burn the image) is not mounted:

```bash
sudo umount /dev/sda*
```

2. If the SSD was mounted, this command will safely unmount it so it's ready for imaging.

3. Navigate to the folder where you have the image (`cd robot_NXUpgrade/`), then run the following command:

```bash
bzip2 -dc g1-nx-j6.2.img.bz2 | sudo dd of=/dev/sda bs=4M status=progress conv=fsync
```

4. After it's done, eject the card with the following commands to safely unplug it:

```bash
sudo sync
sudo udisksctl power-off -b /dev/sda
```

5. **Set the SSD card to the side and proceed with the second part of the flashing process!**

## 4. Put the Robot Into Flashing Mode

1. **Power on the G1** and wait until all three power indicator lights remain steadily lit.

2. **Connect the robot to your laptop/desktop** using a USB-C cable.

3. **Press and hold both white buttons** on the robot at the same time for two seconds.

4. While still holding them, **release the top white button** and continue holding the **bottom button** for 2 seconds until the **three green lights change to two green lights**.

```{image} ../_static/flashing.png
:width: 600px
:align: center
```

5. When only two lights are on, the robot is **now in flashing mode**. Open a new terminal on your computer and enter `lsusb`. You should see text containing `NVIDIA Corp. APX`.

6. You can now proceed to run the following commands:

```bash
sudo tar -xjvf Jetpack_6.2_nx.tar.bz2
cd Jetpack_6.2_nx/Linux_for_Tegra
sudo ./flash_nx_module.sh
```

Wait patiently for about 8 minutes until it shows success.

## 5. Reassemble the Robot

1. After the flashing is complete, **power off the robot**.

2. **Reinstall the Orin NX's NVMe SSD** back into its slot on the G1 robot and secure it with its screw.

3. **Reattach the foam and plastic backing**, using the same tools you used to remove it.

4. **Tighten all screws** to ensure the back cover and handle are securely in place.

5. Turn on `maxn` mode on Jetson Orin using the command: 


```
sudo nvpmodel -m 0
```

and use 


```
sudo jetson_clocks
sudo jetson_clocks --show  
```

to check if it is already in Maxn model. 
