#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from collections import Counter

def load_id_to_label(seg_dir: Path) -> dict:
    j = seg_dir / "id_label_map.json"
    if j.exists():
        raw = json.loads(j.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}
    # fallback básico
    legend = seg_dir / "legend.csv"
    labels = {}
    if legend.exists():
        for i in pd.read_csv(legend)["id"].tolist():
            labels[int(i)] = "background" if i == 0 else ("unknown" if i == 65534 else f"id_{i}")
    return labels

def discover_classes(seg_dir: Path, min_pixels: int) -> list[int]:
    present = Counter()
    for f in sorted(seg_dir.glob("*_stats.csv")):
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            i, p = int(r["id"]), int(r["pixels"])
            if i in (0, 65534) or p < min_pixels:
                continue
            present[i] += p
    return sorted(present.keys())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-pixels", type=int, default=200)
    args = ap.parse_args()

    seg_dir = Path(os.path.expanduser(args.seg_dir))
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    id2label = load_id_to_label(seg_dir)
    class_ids = discover_classes(seg_dir, args.min_pixels)
    classes = [id2label.get(i, f"id_{i}") for i in class_ids]
    (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    id2idx = {cid: i for i, cid in enumerate(class_ids)}

    colors = sorted(seg_dir.glob("*_color.png"))
    ids    = sorted(seg_dir.glob("*_ids.tiff"))
    assert len(colors) == len(ids) > 0

    for cpath, ipath in zip(colors, ids):
        img = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        idm = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED)  # uint16
        H, W = idm.shape[:2]

        # construir máscara de índices (uint8) por clases seleccionadas
        mask = np.zeros((H, W), dtype=np.uint8)
        for obj_id, cls_idx in id2idx.items():
            m = (idm == obj_id)
            mask[m] = cls_idx + 1  # +1 para reservar 0 como 'background' al importar
        # Nota: Roboflow acepta 0 como fondo y valores discretos para clases.
        # Aquí dejamos 0=fondo, 1..N=clases.

        # filtros opcionales por min_pixels: si no hay píxeles para ninguna clase, la máscara queda 0

        # guardar
        cv2.imwrite(str(out_dir / "images" / cpath.name), img)
        cv2.imwrite(str(out_dir / "masks" / (cpath.stem.replace("_color", "") + ".png")), mask)

    print(f"Listo. Export en: {out_dir}")
    print("Sube 'images/' y 'masks/' a Roboflow con el tipo 'Semantic Segmentation (PNG masks)'.")
    
if __name__ == "__main__":
    main()
