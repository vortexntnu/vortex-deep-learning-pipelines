#!/usr/bin/env python3
import sys
import subprocess
import signal
from PyQt5 import QtWidgets, QtCore

# --------- CONFIG ---------
PARENT_FRAME = "camera_rig/Dcam"
CHILD_FRAME  = "camera_rig/segmentation_camera_front"

# Slider ranges (meters / radians)
RANGE_POS_M = 0.50      # +/- 0.50 m
RANGE_ANG_R = 1.57      # +/- ~90deg in rad

# Slider resolution
POS_STEP = 0.001        # 1 mm
ANG_STEP = 0.001        # ~0.057 deg
# --------------------------


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class TfTuner(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TF Tuner (static_transform_publisher)")
        self.proc = None

        self.layout = QtWidgets.QVBoxLayout(self)

        self.info = QtWidgets.QLabel(
            f"Publishing TF:\n  {PARENT_FRAME}  ->  {CHILD_FRAME}\n"
            "Adjust sliders then press 'Adjust' to respawn the publisher."
        )
        self.layout.addWidget(self.info)

        grid = QtWidgets.QGridLayout()
        self.layout.addLayout(grid)

        # Create sliders + spinboxes
        self.controls = {}
        params = [
            ("x (m)", -RANGE_POS_M, RANGE_POS_M, POS_STEP),
            ("y (m)", -RANGE_POS_M, RANGE_POS_M, POS_STEP),
            ("z (m)", -RANGE_POS_M, RANGE_POS_M, POS_STEP),
            ("roll (rad)", -RANGE_ANG_R, RANGE_ANG_R, ANG_STEP),
            ("pitch (rad)", -RANGE_ANG_R, RANGE_ANG_R, ANG_STEP),
            ("yaw (rad)", -RANGE_ANG_R, RANGE_ANG_R, ANG_STEP),
        ]

        for row, (name, lo, hi, step) in enumerate(params):
            lbl = QtWidgets.QLabel(name)
            grid.addWidget(lbl, row, 0)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(int(lo / step))
            slider.setMaximum(int(hi / step))
            slider.setValue(0)
            grid.addWidget(slider, row, 1)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(4)
            spin.setValue(0.0)
            grid.addWidget(spin, row, 2)

            # Sync slider <-> spinbox
            def make_slider_to_spin(s=slider, sp=spin, st=step):
                def _():
                    sp.blockSignals(True)
                    sp.setValue(s.value() * st)
                    sp.blockSignals(False)
                return _
            def make_spin_to_slider(s=slider, sp=spin, st=step):
                def _():
                    s.blockSignals(True)
                    s.setValue(int(sp.value() / st))
                    s.blockSignals(False)
                return _

            slider.valueChanged.connect(make_slider_to_spin())
            spin.valueChanged.connect(make_spin_to_slider())

            self.controls[name] = (slider, spin)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.layout.addLayout(btn_row)

        self.adjust_btn = QtWidgets.QPushButton("Adjust")
        self.adjust_btn.clicked.connect(self.on_adjust)
        btn_row.addWidget(self.adjust_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop)
        btn_row.addWidget(self.stop_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset (0,0,0,0,0,0)")
        self.reset_btn.clicked.connect(self.on_reset)
        btn_row.addWidget(self.reset_btn)

        self.status = QtWidgets.QLabel("Status: idle")
        self.layout.addWidget(self.status)

        self.resize(900, 260)

    def values(self):
        # In the same order ROS expects
        x = self.controls["x (m)"][1].value()
        y = self.controls["y (m)"][1].value()
        z = self.controls["z (m)"][1].value()
        roll  = self.controls["roll (rad)"][1].value()
        pitch = self.controls["pitch (rad)"][1].value()
        yaw   = self.controls["yaw (rad)"][1].value()
        return x, y, z, roll, pitch, yaw

    def spawn_publisher(self, x, y, z, roll, pitch, yaw):
        # Kill previous if running
        self.on_stop()

        cmd = [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "--x", f"{x:.6f}", "--y", f"{y:.6f}", "--z", f"{z:.6f}",
            "--roll", f"{roll:.6f}", "--pitch", f"{pitch:.6f}", "--yaw", f"{yaw:.6f}",
            "--frame-id", PARENT_FRAME,
            "--child-frame-id", CHILD_FRAME,
        ]

        self.status.setText("Status: starting publisher...")
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            self.status.setText("Status: ERROR - 'ros2' not found. Source ROS2 environment first.")
            self.proc = None
            return

        self.status.setText(
            f"Status: running | x={x:.3f} y={y:.3f} z={z:.3f} roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f}"
        )

    def on_adjust(self):
        x, y, z, roll, pitch, yaw = self.values()
        self.spawn_publisher(x, y, z, roll, pitch, yaw)

    def on_stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def on_reset(self):
        for _, (_, spin) in self.controls.items():
            spin.setValue(0.0)
        self.on_adjust()

    def closeEvent(self, event):
        self.on_stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = TfTuner()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
