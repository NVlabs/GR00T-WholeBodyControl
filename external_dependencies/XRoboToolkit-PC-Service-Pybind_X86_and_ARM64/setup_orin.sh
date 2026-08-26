if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    conda install -c conda-forge libstdcxx-ng -y
fi

mkdir -p tmp
cd tmp
git clone -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK 
bash build.sh
cd ../../../..

mkdir -p lib/aarch64
mkdir -p include/aarch64
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/aarch64/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/aarch64/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/aarch64/
# rm -rf tmp

# Build the project
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    conda install -c conda-forge pybind11 -y
else
    pip install pybind11 -y
fi

pip uninstall -y xrobotoolkit_sdk
python setup.py install