# datageneration

Creation of synthetic and labeled point cloud data for testing and AI training purposes.

![](/media/generatedData.gif)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

From `src/` (paths assume `data/` at the repo root):

```bash
python basic.py
python multiple_elements.py
```
