# Cricket Video Analysis with YOLOv5

Frame-by-frame object detection over cricket footage, using a custom-trained YOLOv5 model to locate the player, bat and ball in each frame of a net-practice video.

## How it works

[`cricket.ipynb`](cricket.ipynb) loads a custom YOLOv5 checkpoint through `torch.hub`, opens the video with OpenCV, and iterates frames until the stream ends. For each frame it runs inference and extracts the predicted bounding boxes and class labels from `results.pred[0]`, moving the tensor to CPU and converting to NumPy for downstream geometry work with `shapely`.

```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path=<weights>, force_reload=True)
cap = cv2.VideoCapture(video_path)
results = model(frame)
bboxes = results.pred[0][:, :4].cpu().numpy()
```

The sample clip (`How Virat Kohli plays spin at nets I Indian Cricket Team.mp4`) is committed alongside the notebook.

## Stack

- `torch` + YOLOv5 (via `torch.hub`) — detection
- `opencv-python` — video decoding and frame iteration
- `numpy`, `shapely` — bounding-box geometry

## Running it

```bash
pip install torch opencv-python numpy shapely
jupyter notebook cricket.ipynb
```

## Before it will run

Two things in the notebook are environment-specific and must be changed:

1. **Model weights are not in this repository.** The `path=` argument points to a local YOLOv5 training run (`runs/train/exp20/weights/last.pt`). Supply your own trained checkpoint, or swap to a pretrained model with `torch.hub.load('ultralytics/yolov5', 'yolov5s')`.
2. **Paths are hardcoded to a Windows machine** (`C:\Users\Shriya\...`) for the video, weights and frame-output folder. Replace them with relative paths — the video sits next to the notebook, so `cv2.VideoCapture("How Virat Kohli plays spin at nets I Indian Cricket Team.mp4")` is enough.

## Limitations

- Detection quality depends entirely on the custom checkpoint, which is not published here, so results are not reproducible as committed.
- Inference runs on every frame with no sampling or batching, which is slow on CPU for full-length footage.
- There is no tracking across frames: each detection is independent, so the same object is not given a stable identity over time.
