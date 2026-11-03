import json
import os
import signal
import subprocess
import sys
from typing import Dict

import library.log as log
import library.qtgui as qtgui
from library.universal import G
from PyQt6.QtCore import QProcess, QSettings, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import decoupled_wbc

signal.signal(signal.SIGINT, signal.SIG_DFL)


GR00T_TELEOP_DATA_ROOT = os.path.join(os.path.dirname(decoupled_wbc.__file__), "./external/teleop/data")


class MainWindow(QMainWindow):
    def __init__(
        self,
        fix_pick_up=True,
        fix_pointing=False,
        object_A_options=[
            "small cup",
            "dragon fruit",
            "orange",
            "apple",
            "mango",
            "star fruit",
            "rubiks cube",
            "lemon",
            "lime",
            "red can",
            "blue can",
            "green can",
            "cucumber",
            "bottled water",
            "big cup",
            "mayo",
            "mustard",
            "bok choy",
            "grapes",
            "soup can",
            "mouse",
            "water apple",
            "corn",
            "mug",
            "orange cup",
            "bitter gourd",
            "banana",
            "mangosteen",
            "marker",
            "coffee pod",
            "plastic cup",
            "grapes",
            "small mug",
            "condiment bottles",
            "corn",
            "tools",
            "pear",
            "eggplant",
            "canned beans",
            "potato",
        ],
        object_from_options=[
            "cutting board",
            "pan",
            "plate",
            "bowl",
            "tray",
            "desk",
            "placemat",
            "table",
            "mesh cup",
            "shelf",
        ],
        object_to_options=[
            "cutting board",
            "pan",
            "plate",
            "bowl",
            "tray",
            "microwave",
            "basket",
            "drawer",
            "placemat",
            "clear bin",
            "mesh cup",
            "yellow bin",
            "shelf",
        ],
    ):
        super().__init__()
        # Fix the pick up and place object type
        self.fix_pick_up = fix_pick_up
        self.fix_pointing = fix_pointing

        # window settings
        self.setWindowTitle("Gr00t Capture Test")
        self.setFixedWidth(800)
        self.setMinimumHeight(1000)

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # stop button
        self.stop_button = QPushButton("Emergency Stop", self)
        self.stop_button.setStyleSheet(
            "background-color: red; color: white; font-size: 20px; font-weight: bold; height: 60px;"
        )
        # self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_process)
        self.main_layout.addWidget(self.stop_button)

        # property label
        self.property_label = QLabel("Settings")
        self.property_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #333; max-hedescription.jight: 30px;"
        )
        self.main_layout.addWidget(self.property_label)

        # operator form
        settings_form_layout = QFormLayout()
        self.operator_input = QLineEdit("Zu")
        # self.operator_input.textChanged.connect(self.property_label.setText)
        settings_form_layout.addRow(QLabel("Operator Name:"), self.operator_input)

        # collector form
        self.collector_input = QLineEdit("Zu")
        settings_form_layout.addRow(QLabel("Collector Name:"), self.collector_input)

        # description form
        self.description_input = QLineEdit("3D pick up")

        # object A menu
        self.object_A_input = QComboBox()
        self.object_A_input.addItems(object_A_options)

        # object A menu
        self.object_from_input = QComboBox()
        self.object_from_input.addItems(object_from_options)

        # object B menu
        self.object_to_input = QComboBox()
        self.object_to_input.addItems(object_to_options)

        box_layout = QHBoxLayout()
        box_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        if self.fix_pick_up:
            # box_layout.addWidget(QLabel("Pour Object:"))
            box_layout.addWidget(QLabel("Pick up Object:"))
            box_layout.addWidget(self.object_A_input)
            box_layout.addWidget(QLabel("from:"))
            box_layout.addWidget(self.object_from_input)
            box_layout.addWidget(QLabel("To:"))
            box_layout.addWidget(self.object_to_input)
        elif self.fix_pointing:
            box_layout.addWidget(QLabel("Pointing Object:"))
            box_layout.addWidget(self.object_A_input)
        else:
            box_layout.addWidget(self.description_input)

        # add a button to fix the pick up and place object
        self.save_description_button = QPushButton("Save Description")
        self.save_description_button.setMinimumWidth(100)
        self.save_description_button.clicked.connect(self.save_settings)
        box_layout.addWidget(self.save_description_button)

        settings_form_layout.addRow(QLabel("Task description:"), box_layout)

        # robot form
        self.robot_input = QLineEdit("gr00t002")
        settings_form_layout.addRow(QLabel("Robot Name:"), self.robot_input)

        # vive keyword form
        self.vive_keyword_input = QComboBox()
        self.vive_keyword_input.addItems(["elbow", "knee", "wrist", "shoulder", "foot"])
        settings_form_layout.addRow(QLabel("Vive keyword:"), self.vive_keyword_input)

        settings_form_layout.addRow(QLabel("-" * 300))
        self.vive_ip_input = QLineEdit("192.168.0.182")
        settings_form_layout.addRow(QLabel("Vive ip:"), self.vive_ip_input)

        self.vive_port_input = QLineEdit("5555")
        settings_form_layout.addRow(QLabel("Vive port:"), self.vive_port_input)
        self.manus_port_input = QLineEdit("5556")
        settings_form_layout.addRow(QLabel("Manus port:"), self.manus_port_input)

        # manus form
        # self.manus_input = QLineEdit("foot")
        # settings_form_layout.addRow(QLabel("Manus name:"), self.manus_input)

        self.main_layout.addLayout(settings_form_layout)

        # testing buttons
        self.testing_button_layout = QGridLayout()

        # vive button
        self.test_vive_button = QPushButton("Test Vive")
        self.test_vive_button.setCheckable(True)
        self.test_vive_button.toggled.connect(self.test_vive)
        self.test_vive_button.setMaximumHeight(30)
        self.testing_button_layout.addWidget(self.test_vive_button, 0, 1)
        self.vive_checkbox = QCheckBox("Vive Ready")
        self.testing_button_layout.addWidget(self.vive_checkbox, 0, 2)

        # start manus button
        self.server_manus_button = QPushButton("Start Manus")
        self.server_manus_button.setCheckable(True)
        self.server_manus_button.toggled.connect(self.test_manus_server)
        self.server_manus_button.setMaximumHeight(30)
        self.testing_button_layout.addWidget(self.server_manus_button, 1, 0)

        # manus button
        self.client_manus_button = QPushButton("Test Manus")
        self.client_manus_button.setCheckable(True)
        self.client_manus_button.toggled.connect(self.test_manus_client)
        self.client_manus_button.setMaximumHeight(30)
        self.testing_button_layout.addWidget(self.client_manus_button, 1, 1)

        self.manus_checkbox = QCheckBox("Manus Ready")
        self.manus_checkbox.setEnabled(False)
        self.testing_button_layout.addWidget(self.manus_checkbox, 1, 2)

        # self.testing_button_layout.addWidget()
        self.main_layout.addLayout(self.testing_button_layout)

        container = QWidget()
        container.setLayout(self.main_layout)

        # Set the central widget of the Window.
        self.setCentralWidget(container)

        # Process
        # QProcess for running scripts
        self.process = QProcess(self)
        # self.process.readyRead.connect(self.handle_stdout)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.errorOccurred.connect(self.process_error)
        self.process.finished.connect(self.process_finished)
        self.current_process_name = ""

        # QProcess for running scripts
        self.manus_process = QProcess(self)
        # self.process.readyRead.connect(self.handle_stdout)
        self.manus_process.readyReadStandardOutput.connect(self.handle_stdout)
        self.manus_process.readyReadStandardError.connect(self.handle_stderr)
        self.manus_process.finished.connect(self.process_finished)

        # load history
        self.load_settings()

        # record all buttons
        self.test_buttons: Dict[str, QPushButton] = {
            "test_vive": self.test_vive_button,
            "test_manus": self.client_manus_button,
        }

    def load_settings(self):
        settings = QSettings("Gear", "Teleop")
        self.vive_ip_input.setText(settings.value("vive_ip", "192.168.0.182"))
        self.vive_keyword_input.setCurrentIndex(int(settings.value("vive_keyword", 0)))
        self.robot_input.setText(settings.value("robot_input", ""))
        self.operator_input.setText(settings.value("operator", ""))
        self.collector_input.setText(settings.value("data_collector", ""))
        self.description_input.setText(settings.value("description", ""))
        self.object_A_input.setCurrentIndex(int(settings.value("object", 0)))
        self.object_from_input.setCurrentIndex(int(settings.value("object_from", 0)))
        self.object_to_input.setCurrentIndex(int(settings.value("object_to", 0)))

    def save_settings(self):
        # Create QSettings object (You can provide your app name and organization for persistent storage)
        settings = QSettings("Gear", "Teleop")

        # Save the text from the QLineEdit
        settings.setValue("vive_ip", self.vive_ip_input.text())
        settings.setValue("vive_keyword", self.vive_keyword_input.currentIndex())
        settings.setValue("robot_input", self.robot_input.text())
        settings.setValue("operator", self.operator_input.text())
        settings.setValue("data_collector", self.collector_input.text())

        if self.fix_pick_up:
            description = (
                f"pick {self.object_A_input.currentText()} "
                f"{self.object_from_input.currentText()}->"
                f"{self.object_to_input.currentText()}"
            )
        elif self.fix_pointing:
            description = f"point at {self.object_A_input.currentText()}"
        else:
            description = self.description_input.text()

        settings.setValue("description", description)
        settings.setValue("object", self.object_A_input.currentIndex())
        settings.setValue("object_from", self.object_from_input.currentIndex())
        settings.setValue("object_to", self.object_to_input.currentIndex())

        # Save the settings
        descrition_file = os.path.join(GR00T_TELEOP_DATA_ROOT, "description.json")
        with open(descrition_file, "w") as f:
            json.dump(
                {
                    "operator": self.operator_input.text(),
                    "data_collector": self.collector_input.text(),
                    "description": description,
                },
                f,
            )
        print("Settings saved to {}".format(descrition_file))

    ################################# Process #################################
    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data()
        stdout = data.decode("utf-8")
        G.app.log_window.addLogMessage(stdout, log.DEBUG)

        data = self.manus_process.readAllStandardOutput().data()
        stdout = data.decode("utf-8")
        G.app.log_window.addLogMessage(stdout, log.DEBUG)

    def handle_stderr(self):
        data = self.process.readAllStandardError().data()
        stderr = data.decode("utf-8")
        G.app.log_window.addLogMessage(stderr, log.ERROR)

        data = self.manus_process.readAllStandardError().data()
        stderr = data.decode("utf-8")
        G.app.log_window.addLogMessage(stderr, log.ERROR)

    def process_finished(self, exit_code, exit_status):
        if exit_status == QProcess.ExitStatus.NormalExit:
            G.app.log_window.addLogMessage(
                f"Process finished successfully with exit code {exit_code}.", log.INFO
            )
        else:
            G.app.log_window.addLogMessage(
                f"Process finished with error code {exit_code}.", log.ERROR
            )

        G.app.log_window.addLogMessage("---------------------------------------------", log.INFO)

        # toggle the button for the current task
        print("current_process_name", self.current_process_name, self.test_buttons.keys())
        if self.current_process_name in self.test_buttons.keys():
            if self.test_buttons[self.current_process_name].isChecked():
                self.test_buttons[self.current_process_name].click()

    def process_error(self, error):
        G.app.log_window.addLogMessage(f"Process Error (or stopped manually): {error}", log.ERROR)
        G.app.log_window.addLogMessage("---------------------------------------------", log.INFO)

    ################################# Timer #################################

    def update_log(self):
        print("update log")
        # G.app.log_window.addLogMessage("Update log", log.DEBUG)
        returnBool = self.process.waitForFinished(1000)
        print("returnBool", returnBool)
        if not returnBool:
            data = self.process.readAllStandardOutput().data()
            stdout = data.decode("utf8")
            if stdout:
                G.app.log_window.addLogMessage(stdout, log.DEBUG)
                G.app.log_window.updateView()

    def test_vive(self, checked):
        print("checked", checked)
        if not checked:
            self.test_vive_button.setText("Test Vive")
            self.stop_process()
            self.current_process_name = ""
        else:
            self.current_process_name = "test_vive"
            self.test_vive_button.setText("Stop Test")
            self.stop_button.setEnabled(True)
            G.app.log_window.addLogMessage("Testing Vive", log.DEBUG)
            script = "python"
            args = [
                "decoupled_wbc/control/teleop/main/test_vive.py",
                "--keyword",
                self.vive_keyword_input.currentText(),
                "--ip",
                self.vive_ip_input.text(),
                "--port",
                self.vive_port_input.text(),
            ]
            self.process.start(script, args)

    def test_manus_server(self, checked):
        if not checked:  # 2
            self.server_manus_button.setText("Start Manus")
            self.stop_manus_server()
            self.manus_checkbox.setEnabled(False)
        else:  # 1
            self.server_manus_button.setText("Stop Manus")
            self.manus_checkbox.setEnabled(True)
            self.start_manus_server()

    def start_manus_server(self):
        G.app.log_window.addLogMessage("Starting manus server", log.WARNING)
        script = "python"
        args = ["decoupled_wbc/control/teleop/device/manus.py", "--port", self.manus_port_input.text()]
        self.manus_process.start(script, args)

    def stop_manus_server(self):
        G.app.log_window.addLogMessage("Stopping manus server", log.WARNING)
        self.manus_process.terminate()
        self.manus_process.waitForFinished(2000)
        if self.manus_process.state() != QProcess.ProcessState.NotRunning:
            self.manus_process.kill()
        self.stop_button.setEnabled(False)
        G.app.log_window.addLogMessage("Manus stopped", log.DEBUG)

    def test_manus_client(self, checked):
        if not checked:
            self.client_manus_button.setText("Test Manus")
            self.stop_process()
            self.current_process_name = ""
        else:
            self.current_process_name = "test_manus"
            self.client_manus_button.setText("Stop Test")
            self.stop_button.setEnabled(True)
            G.app.log_window.addLogMessage("Testing", log.DEBUG)
            print("Testing manus")
            script = "python"
            args = [
                "decoupled_wbc/control/teleop/main/test_manus.py",
                "--port",
                self.manus_port_input.text(),
            ]
            self.process.start(script, args)

    ############################### Stop Process #################################

    def stop_process(self):
        G.app.log_window.addLogMessage("Stopping process", log.WARNING)
        # disable all buttons
        for button in self.test_buttons.values():
            button.setCheckable(False)
            button.setChecked(False)
            button.setEnabled(False)

        # kill the process
        if self.process.state() == QProcess.ProcessState.Running:
            # os.kill(self.process.processId(), signal.SIGINT)
            self.process.waitForFinished(2000)  # Wait for up to 2 seconds for termination
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
            # self.stop_button.setEnabled(False)
            G.app.log_window.addLogMessage("Script stopped", log.DEBUG)

        if self.current_process_name in ["test_oak", "record_demonstration", "test_robot"]:
            command = (
                f"ps aux | grep {self.current_process_name}"
                + " | grep -v grep | awk '{print $2}' | xargs kill"
            )

            # Execute the command with subprocess.Popen, capturing both stdout and stderr
            sub_process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Get the output and errors (if any)
            stdout, stderr = sub_process.communicate()
            # Ensure the process is successfully terminated
            sub_process.wait()

        # enable all buttons
        for button in self.test_buttons.values():
            button.setCheckable(True)
            button.setEnabled(True)

        # check the button if the button was checked
        if self.current_process_name in self.test_buttons.keys():
            if self.test_buttons[self.current_process_name].isChecked():
                self.test_buttons[self.current_process_name].setChecked(False)


class LogWindow(qtgui.ListView):
    def __init__(self):
        super(LogWindow, self).__init__()
        self.level = log.DEBUG
        self.allowMultipleSelection(True)

    def setLevel(self, level):
        self.level = level
        self.updateView()

    def keyPressEvent(self, e):
        e.ignore()

    def updateView(self):
        for i in range(self.count()):
            ilevel = self.getItemData(i)
            self.showItem(i, ilevel >= self.level)
            self.setItemColor(i, log.getLevelColor(ilevel))

    def addLogMessage(self, text, level=log.INFO):
        index = self.count()
        color = log.getLevelColor(level)
        self.addLogItem(text, color, level)
        self.showItem(index, level >= self.level)


class GCApplication(QApplication):
    def __init__(self, argv):
        super(GCApplication, self).__init__(argv)

        # application
        if G.app is not None:
            raise RuntimeError("MHApplication is a singleton")
        G.app = self

        self.mainwin = MainWindow()
        self.log_window = LogWindow()
        self.mainwin.main_layout.addWidget(self.log_window)
        self.mainwin.show()
        # self.log_window.show()


# Function to terminate the subprocess gracefully
def terminate_subprocess(proc):
    if proc.poll() is None:  # Check if the process is still running
        proc.terminate()  # Terminate the process
        print("Subprocess terminated.")


if __name__ == "__main__":
    app = GCApplication(sys.argv)
    sys.exit(app.exec())
