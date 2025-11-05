#!/usr/bin/env python3
import argparse, json, os, re, shutil
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

def load_id_to_label(seg_dir: Path) -> dict:
    # 1) JSON explícito, 2) CSV (id,label), 3) fallback: reglas por defecto
    json_path = seg_dir / "id_label_map.json"
    csv_path  = seg_dir / "id_label_map.csv"
    if json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}
    if csv_path.exists():
        m = {}
        for r in pd.read_csv(csv_path).to_dict("records"):
            m[int(r["id"])] = str(r["label"])
        return m
    # Fallback: inferir de legend.csv o del propio dataset cuando no hay mapa
    m = {}
    legend = seg_dir / "legend.csv"
    if legend.exists():
        ids = pd.read_csv(legend)["id"].tolist()
        for i in ids:
            if i == 0:
                m[i] = "background"
            elif i == 65534:
                m[i] = "unknown"
            else:
                m[i] = f"id_{i}"
    return m

def yolo_line(cx, cy, w, h, img_w, img_h, cls_idx):
    return f"{cls_idx} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True, help="Carpeta con *_color.png, *_ids.tiff, legend.csv")
    ap.add_argument("--out-dir", required=True, help="Salida YOLO")
    ap.add_argument("--min-pixels", type=int, default=200, help="Mínimo de píxeles por objeto")
    args = ap.parse_args()

    seg_dir = Path(os.path.expanduser(args.seg_dir))
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Cargar mapeo id->label
    id2label = load_id_to_label(seg_dir)

    color_files = sorted(seg_dir.glob("*_color.png"))
    id_files    = sorted(seg_dir.glob("*_ids.tiff"))
    assert len(color_files) == len(id_files) and len(color_files) > 0, "Dataset incompleto"

    # Descubrir ids presentes (para clases.txt ordenadas)
    present_ids = set()
    for f in sorted(seg_dir.glob("*_stats.csv")):
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            obj_id = int(r["id"])
            if obj_id in (0, 65534):  # background y unknown fuera
                continue
            if int(r["pixels"]) >= args.min_pixels:
                present_ids.add(obj_id)
    # Orden estable por ID
    class_ids = sorted(present_ids)
    # Etiquetas (si falta en id2label, crear por defecto)
    classes = []
    for obj_id in class_ids:
        label = id2label.get(obj_id, f"id_{obj_id}")
        classes.append(label)

    # Guardar clases
    (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    # Mapa id→índice YOLO (0..N-1)
    id2idx = {cid: i for i, cid in enumerate(class_ids)}

    # Convertir cada frame
    for cpath, ipath in zip(color_files, id_files):
        img = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        ids = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED)  # uint16 por píxel
        assert ids is not None and img is not None, f"Error con {cpath.name}"

        H, W = ids.shape[:2]
        # contar píxeles por id rápidamente
        flat = ids.reshape(-1)
        uniq, counts = np.unique(flat, return_counts=True)

        lines = []
        for obj_id, count in zip(uniq, counts):
            obj_id = int(obj_id)
            if obj_id in (0, 65534):
                continue
            if count < args.min_pixels:
                continue
            if obj_id not in id2idx:
                # id presente pero filtrado globalmente; se ignora
                continue

            mask = (ids == obj_id)
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            w = (x_max - x_min + 1)
            h = (y_max - y_min + 1)
            cx = x_min + w / 2
            cy = y_min + h / 2

            cls_idx = id2idx[obj_id]
            lines.append(yolo_line(cx, cy, w, h, W, H, cls_idx))

        # escribir label
        label_path = (out_dir / "labels" / (cpath.stem.replace("_color", "") + ".txt"))
        label_path.write_text("\n".join(lines), encoding="utf-8")

        # copiar imagen
        shutil.copy2(cpath, out_dir / "images" / cpath.name)

    # data.yaml mínimo para YOLO/Roboflow
    yaml = [
        f"path: {out_dir.resolve()}",
        "train: images",
        "val: images",  # si no separas, Roboflow redivide al importar
        f"names: {classes}"
    ]
    (out_dir / "data.yaml").write_text("\n".join(yaml), encoding="utf-8")
    print(f"Listo. Export en: {out_dir}")

if __name__ == "__main__":
    main()
