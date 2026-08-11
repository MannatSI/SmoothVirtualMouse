# SmoothVirtualMouse
Smooth hand-gesture virtual mouse using MediaPipe, OpenCV and Python.
# Smooth Virtual Mouse

A Python-based virtual mouse controlled using hand gestures.

The project uses MediaPipe Hand Landmarker, OpenCV and PyAutoGUI to control the computer mouse using a webcam.

## Features

* ☝️ Index finger — move cursor
* 🤏 Index + thumb — left click
* 🤏 Middle + thumb — right click
* ✊ Fist — drag
* 🖐️ Open palm — scroll
* ESC — exit

## Requirements

* Windows
* Python 3.13
* Webcam
* Internet connection for installing Python packages

Recommended webcam resolution:

* 640×480 or higher

## Installation

Open Command Prompt in the project folder.

Check Python:

```bash
py -3.13 --version
```

You should see:

```text
Python 3.13.x
```

Create a virtual environment:

```bash
py -3.13 -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run

Make sure these files are together:

```text
SmoothVirtualMouse/
├── mouse.py
├── hand_landmarker.task
└── requirements.txt
```

Then:

```bash
python mouse.py
```

## Controls

| Gesture              | Action      |
| -------------------- | ----------- |
| ☝️ Index finger only | Move cursor |
| 🤏 Index + thumb     | Left click  |
| 🤏 Middle + thumb    | Right click |
| ✊ Fist               | Drag        |
| 🖐️ Open palm        | Scroll      |
| ESC                  | Exit        |

## Tips

For better tracking:

* Use good lighting.
* Keep your hand approximately 30–60 cm from the webcam.
* Keep your index finger clearly visible.
* Avoid very dark backgrounds.
* Keep the webcam stable.
* Stay inside the yellow control area.

## Troubleshooting

### Camera doesn't open

Check that Windows has camera permission enabled and that another application isn't using the webcam.

### MediaPipe error

Make sure the virtual environment is activated and run:

```bash
python -m pip install -r requirements.txt
```

### Model not found

Make sure:

```text
hand_landmarker.task
```

is in the same directory as:

```text
mouse.py
```

### Check installed packages

Run:

```bash
python -m pip list
```

## License

Use and modify this project for personal and educational purposes.
