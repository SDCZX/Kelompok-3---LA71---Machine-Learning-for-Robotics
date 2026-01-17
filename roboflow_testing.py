import roboflow
import os

#Download Dataset dari Roboflow
rf = roboflow.Roboflow(api_key="T3OT6xsuMPjc9ioarDRr")
workspace = rf.workspace("ignatius-bryan-oden")
project = workspace.project("trash-identifier-nnvui")
dataset = project.version(12)
dataset_location = dataset.download("yolov5")

#Path ke data.yaml
yaml_path = os.path.join(dataset_location.location, "data.yaml")

# --- Camera demo that separates organic vs inorganic detections ---
import glob
import argparse
import sys
from typing import Optional
import time
import torch


def find_best_weights():
    """Return the most recent trained weights (best.pt or last.pt) or a fallback yolov5s.pt."""
    candidates = glob.glob(os.path.join("runs", "train", "*", "weights", "best.pt")) + glob.glob(os.path.join("runs", "train", "*", "weights", "last.pt"))
    candidates = [c for c in candidates if os.path.isfile(c)]
    candidates.sort(key=os.path.getmtime, reverse=True)
    if candidates:
        return candidates[0]
    # fallback to packaged weights
    fallback = os.path.join("yolov5", "yolov5s.pt")
    if os.path.isfile(fallback):
        return fallback
    return None


def camera_separate_demo(weights: Optional[str] = None, source=0, imgsz=(640, 640), conf_thres: float = 0.25, device: str = "", save_crops: bool = False):
    """Webcam demo that groups detections into Organic vs Inorganic and overlays counts on-screen.

    - Treats class name 'organic' as Organic; all other classes are considered Inorganic (e.g., 'inorganic').
    - Optionally saves crops per category under runs/detect/camera_separate/{organic,inorganic}/
    """
    # Late imports from yolov5 utils (keeps script independent for unit tests)
    try:
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.dataloaders import LoadStreams
        from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
        from yolov5.utils.torch_utils import select_device
        from ultralytics.utils.plotting import Annotator, save_one_box
    except Exception:
        # try adding local yolov5 folder to sys.path
        sys.path.append(os.path.join(os.path.dirname(__file__), "yolov5"))
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.dataloaders import LoadStreams
        from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
        from yolov5.utils.torch_utils import select_device
        from ultralytics.utils.plotting import Annotator, save_one_box

    if weights is None:
        weights = find_best_weights()
    if weights is None:
        raise FileNotFoundError("No weights found. Place 'best.pt' in runs/train/*/weights or include 'yolov5/yolov5s.pt'.")

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadStreams(str(source), img_size=imgsz, stride=stride, auto=pt)

    save_root = os.path.join("runs", "detect", "camera_separate")
    organic_dir = os.path.join(save_root, "organic")
    inorganic_dir = os.path.join(save_root, "inorganic")
    if save_crops:
        os.makedirs(organic_dir, exist_ok=True)
        os.makedirs(inorganic_dir, exist_ok=True)

    # Normalize model class names into a list of strings
    if isinstance(names, dict):
        names_list = list(names.values())
    else:
        names_list = list(names)
    names_list = [str(n) for n in names_list]
    print(f"Model classes: {names_list}")
    has_organic_label = any(n.lower() == "organic" for n in names_list)

    # Default COCO-ish mapping (substring matching) when 'organic' label is not present
    default_map = {
        # all training from COCO Roboflow
        "apple": "Organic",
        "banana": "Organic",
        "orange": "Organic",
        "food": "Organic",
        "banana": "Organic",
        "fruit": "Organic",
        "vegetable": "Organic",
        "leaf": "Organic",
        "bread": "Organic",
        "meat": "Organic",
        # all training from COCO Roboflow
        "bottle": "Inorganic",
        "cup": "Inorganic",
        "book": "Inorganic",
        "refrigerator": "Inorganic",
        "chair": "Inorganic",
        "couch": "Inorganic",
        "bowl": "Inorganic",
        "knife": "Inorganic",
        "fork": "Inorganic",
        "spoon": "Inorganic",
        "cell phone": "Inorganic",
        "phone": "Inorganic",
        "plastic": "Inorganic",
        "metal": "Inorganic",
        "can": "Inorganic",
        "glass": "Inorganic",
        "keyboard": "Inorganic",
        "laptop": "Inorganic",
        "tv": "Inorganic",
        "remote": "Inorganic",
    }

    def map_label_to_category(label: str):
        """Map a model label to 'Organic'/'Inorganic' or return None to ignore."""
        l = label.lower()
        if has_organic_label:
            # If model has an 'organic' class, use that as organic and everything else inorganic
            if l == "organic":
                return "Organic"
            else:
                return "Inorganic"
        # else try substring matching in default_map
        for k, v in default_map.items():
            if k in l:
                return v
        return None

    print(f"Mapping strategy: {'model-based' if has_organic_label else 'default mapping (COCO)'}")

    print(f"Using weights: {weights} | source: {source} | imgsz: {imgsz}")

    for path, im, im0s, vid_cap, s in dataset:
        # Preprocess
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        # Inference
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if isinstance(im0s, list) else im0s.copy()
            annotator = Annotator(im0, line_width=2, example=str(names))

            organic_count = 0
            inorganic_count = 0

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Filter and process detections
                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    label = str(names[cls])
                    category = map_label_to_category(label)
                    if category is None:
                        # Ignore this detection (not mapped to organic/inorganic)
                        continue

                    if category == "Organic":
                        organic_count += 1
                        color = (0, 200, 0)  # green
                        if save_crops:
                            save_one_box(xyxy, im0, file=os.path.join(organic_dir, f"crop_{int(time.time()*1000)}.jpg"), BGR=True)
                    else:
                        inorganic_count += 1
                        color = (0, 0, 200)  # red
                        if save_crops:
                            save_one_box(xyxy, im0, file=os.path.join(inorganic_dir, f"crop_{int(time.time()*1000)}.jpg"), BGR=True)

                    # Draw a simplified label showing only category and confidence
                    label_text = f"{category} {conf:.2f}"
                    annotator.box_label(xyxy, label_text, color=color)

            # Overlay counts
            info_text = f"Organic: {organic_count}   Inorganic: {inorganic_count}"
            cv2 = __import__("cv2")
            cv2.putText(im0, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Show image
            cv2.imshow("camera_separate", im0)
            if cv2.waitKey(1) & 0xFF == ord("c"):
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo: separates organic vs inorganic detections")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file (optional)")
    parser.add_argument("--source", type=str, default="0", help="Camera source (default 0)")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Inference size h w or h (square)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="", help="CUDA device, e.g., '0' or 'cpu'")
    parser.add_argument("--save-crops", action="store_true", help="Save detected crops into runs/detect/camera_separate/{organic,inorganic}")
    args = parser.parse_args()
    imgsz = tuple(args.imgsz) if len(args.imgsz) > 1 else (args.imgsz[0], args.imgsz[0])
    camera_separate_demo(weights=args.weights, source=args.source, imgsz=imgsz, conf_thres=args.conf_thres, device=args.device, save_crops=args.save_crops)