import os
import glob
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ---- CONFIG ----
CLASS_ID = 0
NUM_KEYPOINTS_PER_OBJECT = 2   # number of keypoints per object
KEYPOINT_VISIBILITY = 2        # v=2 (visible)

# Zoom config
ZOOM_STEP = 1.1
ZOOM_MIN = 0.05
ZOOM_MAX = 20.0
CLICK_DRAG_THRESHOLD_PX = 4  # movement beyond this counts as pan, not a click

# OBB drag config
POINT_HIT_RADIUS_PX = 10  # how close you need to click to "grab" a corner point


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def norm_xy(x, y, w, h):
    # normalize pixel coords to [0,1] (YOLO-style)
    return x / w, y / h


# ----------------- Geometry helpers -----------------
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _mul(a, s):
    return (a[0] * s, a[1] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _len(a):
    return math.hypot(a[0], a[1])


def _normalize(v):
    L = _len(v)
    if L <= 1e-12:
        return (1.0, 0.0)
    return (v[0] / L, v[1] / L)


def _rotate_point(p, center, ang_rad):
    s = math.sin(ang_rad)
    c = math.cos(ang_rad)
    x = p[0] - center[0]
    y = p[1] - center[1]
    xr = x * c - y * s
    yr = x * s + y * c
    return (xr + center[0], yr + center[1])


def _poly_center(poly):
    # simple average of vertices
    x = sum(p[0] for p in poly) / len(poly)
    y = sum(p[1] for p in poly) / len(poly)
    return (x, y)


def _on_segment(p, a, b, eps=1e-9):
    # collinear and within bounds
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(a, b, c, d):
    # Proper segment intersection with collinear handling
    ab = _sub(b, a)
    ac = _sub(c, a)
    ad = _sub(d, a)
    cd = _sub(d, c)
    ca = _sub(a, c)
    cb = _sub(b, c)

    d1 = _cross(ab, ac)
    d2 = _cross(ab, ad)
    d3 = _cross(cd, ca)
    d4 = _cross(cd, cb)

    eps = 1e-9

    # general case
    if (d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps):
        if (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps):
            return True

    # collinear cases
    if abs(d1) <= eps and _on_segment(c, a, b): return True
    if abs(d2) <= eps and _on_segment(d, a, b): return True
    if abs(d3) <= eps and _on_segment(a, c, d): return True
    if abs(d4) <= eps and _on_segment(b, c, d): return True

    return False


def _point_in_poly(pt, poly):
    # ray casting
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def _segment_intersects_poly(seg_a, seg_b, poly):
    # If segment crosses any edge or lies inside polygon
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        if _segments_intersect(seg_a, seg_b, p1, p2):
            return True
    if _point_in_poly(seg_a, poly) or _point_in_poly(seg_b, poly):
        return True
    return False


def _angle_of(v):
    return math.atan2(v[1], v[0])


def _wrap_pi(a):
    # wrap to (-pi, pi]
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Labeling Tool")

        # State
        self.image_paths = []
        self.idx = 0
        self.current_image = None
        self.tk_image = None

        # View transform (image -> canvas): zoom + pan
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # For click vs drag
        self._mouse_down = False
        self._dragging = False
        self._down_x = 0
        self._down_y = 0
        self._last_x = 0
        self._last_y = 0

        # OBB point dragging state
        # (kind, box_index, point_index) where kind in {"obb_done","obb_cur"}
        self._drag_point = None

        # Guide line tool ("g"): two points, not saved
        self.guide_mode = False
        self.guide_points = []  # image coords [(x,y), (x,y)] once complete

        # Modes
        self.mode = tk.StringVar(value="obb")  # "obb" or "kpt"

        # Multiple OBBs: list of boxes, each is 4 (x,y) points
        self.obbs = []            # list[list[tuple(x,y)]]
        self.obb_current = []     # list[tuple(x,y)] points being drawn (0..4)

        # Multiple keypoint objects: each object has NUM_KEYPOINTS_PER_OBJECT points
        self.kpt_sets = []        # list[list[tuple(x,y)]]
        self.kpt_current = []     # list[tuple(x,y)] points being drawn (0..NUM_KEYPOINTS_PER_OBJECT)

        # UI
        self._build_ui()
        self.root.bind("<KeyPress>", self.on_key)

    # ----------------- Paths / IO -----------------
    def base_dir(self):
        if not self.image_paths:
            return None
        return os.path.dirname(self.image_paths[0])

    def obb_dir(self):
        base = self.base_dir()
        if not base:
            return None
        out = os.path.join(base, "obb_label")
        os.makedirs(out, exist_ok=True)
        return out

    def kpt_dir(self):
        base = self.base_dir()
        if not base:
            return None
        out = os.path.join(base, "keypoint_label")
        os.makedirs(out, exist_ok=True)
        return out

    def stem(self):
        img_path = self.image_paths[self.idx]
        return os.path.splitext(os.path.basename(img_path))[0]

    def obb_label_path(self):
        out = self.obb_dir()
        if not out:
            return None
        return os.path.join(out, self.stem() + ".txt")

    def kpt_label_path(self):
        out = self.kpt_dir()
        if not out:
            return None
        return os.path.join(out, self.stem() + ".txt")

    def load_existing_labels(self):
        # Reset current in-memory labels
        self.obbs = []
        self.obb_current = []
        self.kpt_sets = []
        self.kpt_current = []
        self.guide_points = []
        self.guide_mode = False
        self._drag_point = None

        if self.current_image is None:
            return

        iw, ih = self.current_image.size

        # ---- load OBBs ----
        obb_path = self.obb_label_path()
        if obb_path and os.path.exists(obb_path):
            try:
                with open(obb_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 1 + 8:
                            continue
                        nums = list(map(float, parts[1:]))
                        pts = []
                        for i in range(0, 8, 2):
                            x = clamp(nums[i] * iw, 0, iw)
                            y = clamp(nums[i + 1] * ih, 0, ih)
                            pts.append((x, y))
                        if len(pts) == 4:
                            self.obbs.append(pts)
            except Exception:
                pass

        # ---- load Keypoints ----
        kpt_path = self.kpt_label_path()
        if kpt_path and os.path.exists(kpt_path):
            try:
                with open(kpt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 1 + 4 + 3 * NUM_KEYPOINTS_PER_OBJECT:
                            continue
                        kp_parts = parts[5:]
                        kps = []
                        ok = True
                        for i in range(NUM_KEYPOINTS_PER_OBJECT):
                            j = 3 * i
                            try:
                                x_n = float(kp_parts[j + 0])
                                y_n = float(kp_parts[j + 1])
                            except Exception:
                                ok = False
                                break
                            kps.append((clamp(x_n * iw, 0, iw), clamp(y_n * ih, 0, ih)))
                        if ok and len(kps) == NUM_KEYPOINTS_PER_OBJECT:
                            self.kpt_sets.append(kps)
            except Exception:
                pass

    # ----------------- UI -----------------
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="Prev (P)", command=self.prev_image).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="Next (N)", command=self.next_image).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="Save (S)", command=self.save_labels).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Button(top, text="Undo (U)", command=self.undo).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="Delete Obj (D)", command=self.delete_last_object).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Button(top, text="Clear (C)", command=self.clear_current).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(top, text="Reset View (R)", command=self.reset_view).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(top, text="Mode:").pack(side=tk.LEFT, padx=(12, 4))
        tk.Radiobutton(top, text="OBB (O)", variable=self.mode, value="obb", command=self.redraw).pack(side=tk.LEFT)
        tk.Radiobutton(
            top,
            text=f"Keypoints (K) [{NUM_KEYPOINTS_PER_OBJECT}/obj]",
            variable=self.mode,
            value="kpt",
            command=self.redraw
        ).pack(side=tk.LEFT)

        self.status = tk.Label(top, text="No folder loaded", anchor="w")
        self.status.pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(self.root, bg="#222222", width=1000, height=700)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Mouse interactions
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Mouse wheel zoom (Windows/macOS)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        # Mouse wheel zoom (Linux)
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)

        self.canvas.bind("<Configure>", lambda e: self.redraw())

    # ----------------- Image navigation -----------------
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder:
            return

        paths = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
        paths = sorted(set(paths))

        if not paths:
            messagebox.showerror("No images", "No supported images found in the selected folder.")
            return

        self.image_paths = paths
        self.idx = 0
        self.load_image()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw()

    def load_image(self):
        if not self.image_paths:
            return
        img_path = self.image_paths[self.idx]
        self.current_image = Image.open(img_path).convert("RGB")
        _ = self.obb_dir()
        _ = self.kpt_dir()
        self.load_existing_labels()
        self.reset_view()
        self.redraw()

    def next_image(self):
        if self.image_paths and self.idx < len(self.image_paths) - 1:
            self.idx += 1
            self.load_image()

    def prev_image(self):
        if self.image_paths and self.idx > 0:
            self.idx -= 1
            self.load_image()

    # ----------------- View transform helpers -----------------
    def base_fit_transform(self):
        """Compute base fit-to-canvas scale and top-left offset (without zoom/pan)."""
        if self.current_image is None:
            return None

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.current_image.size

        s = min(cw / iw, ch / ih)
        disp_w = iw * s
        disp_h = ih * s
        ox = (cw - disp_w) / 2.0
        oy = (ch - disp_h) / 2.0
        return s, ox, oy, disp_w, disp_h

    def image_to_canvas(self, x, y):
        base = self.base_fit_transform()
        if base is None:
            return 0, 0
        s, ox, oy, _, _ = base
        cx = ox + (x * s) * self.zoom + self.pan_x
        cy = oy + (y * s) * self.zoom + self.pan_y
        return cx, cy

    def canvas_to_image(self, cx, cy):
        base = self.base_fit_transform()
        if base is None:
            return 0, 0
        s, ox, oy, _, _ = base
        x = (cx - ox - self.pan_x) / (s * self.zoom)
        y = (cy - oy - self.pan_y) / (s * self.zoom)
        return x, y

    # ----------------- Drawing -----------------
    def redraw(self):
        self.canvas.delete("all")
        if self.current_image is None:
            return

        base = self.base_fit_transform()
        if base is None:
            return
        s, ox, oy, _, _ = base

        iw, ih = self.current_image.size
        draw_w = max(1, int(iw * s * self.zoom))
        draw_h = max(1, int(ih * s * self.zoom))
        img_disp = self.current_image.resize((draw_w, draw_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(img_disp)

        self.canvas.create_image(ox + self.pan_x, oy + self.pan_y, anchor="nw", image=self.tk_image)

        self.draw_obbs()
        self.draw_keypoints()
        self.draw_guide()

        img_path = self.image_paths[self.idx]
        obb_saved = os.path.exists(self.obb_label_path() or "")
        kpt_saved = os.path.exists(self.kpt_label_path() or "")
        guide_txt = "GUIDE: ON (click 2 pts)" if self.guide_mode else "GUIDE: off (press G)"
        self.status.config(
            text=(
                f"[{self.idx+1}/{len(self.image_paths)}] {os.path.basename(img_path)} | "
                f"mode={self.mode.get()} | zoom={self.zoom:.2f} | {guide_txt} | "
                f"obbs={len(self.obbs)} (+{len(self.obb_current)}pts) | "
                f"kpt_objs={len(self.kpt_sets)} (+{len(self.kpt_current)}pts) | "
                f"saved: obb={'Y' if obb_saved else 'N'}, kpt={'Y' if kpt_saved else 'N'}"
            )
        )

    def draw_obbs(self):
        # Completed OBBs
        for b_idx, obb in enumerate(self.obbs):
            pts_canvas = []
            for i, (x, y) in enumerate(obb):
                cx, cy = self.image_to_canvas(x, y)
                pts_canvas.extend([cx, cy])
                r = 4
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="cyan", width=2)
                self.canvas.create_text(cx + 10, cy, text=f"B{b_idx}P{i+1}", fill="cyan", anchor="w")
            if len(pts_canvas) == 8:
                self.canvas.create_polygon(*pts_canvas, outline="cyan", fill="", width=2)

        # Current OBB being built
        if self.obb_current:
            pts_canvas = []
            for i, (x, y) in enumerate(self.obb_current):
                cx, cy = self.image_to_canvas(x, y)
                pts_canvas.extend([cx, cy])
                r = 4
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="cyan", width=2)
                self.canvas.create_text(cx + 10, cy, text=f"CUR{i+1}", fill="cyan", anchor="w")
            if len(pts_canvas) >= 4:
                self.canvas.create_line(*pts_canvas, fill="cyan", width=2)

    def draw_keypoints(self):
        for o_idx, kps in enumerate(self.kpt_sets):
            for i, (x, y) in enumerate(kps):
                cx, cy = self.image_to_canvas(x, y)
                r = 4
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="yellow", width=2)
                self.canvas.create_text(cx + 10, cy, text=f"O{o_idx}K{i}", fill="yellow", anchor="w")

        for i, (x, y) in enumerate(self.kpt_current):
            cx, cy = self.image_to_canvas(x, y)
            r = 4
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="yellow", width=2)
            self.canvas.create_text(cx + 10, cy, text=f"CURK{i}", fill="yellow", anchor="w")

    def draw_guide(self):
        # Guide points / line (not saved)
        if not self.guide_points:
            return
        # draw points
        for i, (x, y) in enumerate(self.guide_points):
            cx, cy = self.image_to_canvas(x, y)
            r = 5
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="magenta", width=2)
            self.canvas.create_text(cx + 12, cy, text=f"G{i+1}", fill="magenta", anchor="w")

        # draw line if we have 2 points
        if len(self.guide_points) == 2:
            (x1, y1), (x2, y2) = self.guide_points
            c1 = self.image_to_canvas(x1, y1)
            c2 = self.image_to_canvas(x2, y2)
            self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill="magenta", width=2)

    # ----------------- OBB drag hit-test -----------------
    def find_nearest_obb_point(self, cx, cy):
        """
        Returns (kind, box_index, point_index) if near a point, else None.
        kind: "obb_done" or "obb_cur"
        """
        best = None
        best_d2 = (POINT_HIT_RADIUS_PX ** 2)

        # current points
        for pi, (x, y) in enumerate(self.obb_current):
            px, py = self.image_to_canvas(x, y)
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = ("obb_cur", -1, pi)

        # completed boxes
        for bi, obb in enumerate(self.obbs):
            for pi, (x, y) in enumerate(obb):
                px, py = self.image_to_canvas(x, y)
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = ("obb_done", bi, pi)

        return best

    # ----------------- Mouse interactions: pan vs click vs drag-point -----------------
    def on_mouse_down(self, event):
        self._mouse_down = True
        self._dragging = False
        self._down_x, self._down_y = event.x, event.y
        self._last_x, self._last_y = event.x, event.y
        self._drag_point = None

        # If in OBB mode, allow dragging of existing OBB corner points
        if self.mode.get() == "obb" and not self.guide_mode:
            hit = self.find_nearest_obb_point(event.x, event.y)
            if hit is not None:
                self._drag_point = hit
                # start drag immediately; do not pan
                self._dragging = True

    def on_mouse_drag(self, event):
        if not self._mouse_down:
            return

        # Dragging an OBB corner point
        if self._drag_point is not None:
            x, y = self.canvas_to_image(event.x, event.y)
            iw, ih = self.current_image.size
            x = float(clamp(x, 0, iw))
            y = float(clamp(y, 0, ih))

            kind, bi, pi = self._drag_point
            if kind == "obb_cur":
                if 0 <= pi < len(self.obb_current):
                    self.obb_current[pi] = (x, y)
            else:
                if 0 <= bi < len(self.obbs) and 0 <= pi < len(self.obbs[bi]):
                    self.obbs[bi][pi] = (x, y)

            self.redraw()
            return

        # Otherwise, pan like before (click+drag)
        dx = event.x - self._last_x
        dy = event.y - self._last_y

        if not self._dragging:
            mdx = event.x - self._down_x
            mdy = event.y - self._down_y
            if (mdx * mdx + mdy * mdy) >= (CLICK_DRAG_THRESHOLD_PX * CLICK_DRAG_THRESHOLD_PX):
                self._dragging = True

        if self._dragging:
            self.pan_x += dx
            self.pan_y += dy
            self._last_x, self._last_y = event.x, event.y
            self.redraw()

    def on_mouse_up(self, event):
        if not self._mouse_down:
            return
        self._mouse_down = False

        # If we were dragging a point, finish drag and do not click-add
        if self._drag_point is not None:
            self._drag_point = None
            return

        # If we were panning, do not click-add
        if self._dragging:
            self._dragging = False
            return

        # Otherwise it's a click
        self.on_click(event)

    def on_click(self, event):
        if self.current_image is None:
            return

        x, y = self.canvas_to_image(event.x, event.y)
        iw, ih = self.current_image.size

        # Ignore clicks outside image bounds
        if x < 0 or y < 0 or x > iw or y > ih:
            return

        x = float(clamp(x, 0, iw))
        y = float(clamp(y, 0, ih))

        # Guide mode: take two points, do NOT save; after 2nd point, apply alignment
        if self.guide_mode:
            self.guide_points.append((x, y))
            if len(self.guide_points) == 2:
                self.apply_parallel_to_guide()
                # turn off guide mode after application
                self.guide_mode = False
                # keep line visible for reference, but you asked "never saved" (it isn't)
                # If you'd rather it disappear immediately, uncomment:
                # self.guide_points = []
            self.redraw()
            return

        # Normal annotation click
        if self.mode.get() == "obb":
            self.obb_current.append((x, y))
            if len(self.obb_current) == 4:
                self.obbs.append(self.obb_current)
                self.obb_current = []
        else:
            self.kpt_current.append((x, y))
            if len(self.kpt_current) == NUM_KEYPOINTS_PER_OBJECT:
                self.kpt_sets.append(self.kpt_current)
                self.kpt_current = []

        self.redraw()

    # ----------------- Guide tool: make OBBs parallel -----------------
    def apply_parallel_to_guide(self):
        """
        Uses the guide line direction. ONLY adjusts OBBs that INTERSECT the guide segment.
        Rotates each such OBB around its center so that one of its edges becomes parallel
        to the guide direction.
        """
        if len(self.guide_points) != 2:
            return
        g1, g2 = self.guide_points
        gdir = _normalize(_sub(g2, g1))
        g_ang = _angle_of(gdir)

        changed = 0
        for bi, obb in enumerate(self.obbs):
            if len(obb) != 4:
                continue

            # ✅ Only change boxes that intersect the guide segment
            if not _segment_intersects_poly(g1, g2, obb):
                continue

            # Pick which edge to align: edge01 or edge12
            e01 = _sub(obb[1], obb[0])
            e12 = _sub(obb[2], obb[1])

            a01 = _angle_of(_normalize(e01))
            a12 = _angle_of(_normalize(e12))

            # Parallel means angle difference ≈ 0 or ≈ pi
            def ang_dist(a, b):
                d = _wrap_pi(a - b)
                return min(abs(d), abs(_wrap_pi(d - math.pi)), abs(_wrap_pi(d + math.pi)))

            d01 = ang_dist(a01, g_ang)
            d12 = ang_dist(a12, g_ang)

            src_ang = a01 if d01 <= d12 else a12

            # Compute smallest rotation to match guide (allow 180° flip)
            raw = _wrap_pi(g_ang - src_ang)
            candidates = [raw, _wrap_pi(raw + math.pi), _wrap_pi(raw - math.pi)]
            rot = min(candidates, key=lambda t: abs(t))

            center = _poly_center(obb)
            new_obb = [_rotate_point(p, center, rot) for p in obb]
            self.obbs[bi] = new_obb
            changed += 1


    # ----------------- Zoom handling -----------------
    def zoom_at(self, canvas_x, canvas_y, factor):
        old_zoom = self.zoom
        new_zoom = clamp(old_zoom * factor, ZOOM_MIN, ZOOM_MAX)
        if new_zoom == old_zoom:
            return

        ix, iy = self.canvas_to_image(canvas_x, canvas_y)
        self.zoom = new_zoom

        base = self.base_fit_transform()
        if base is None:
            return
        s, ox, oy, _, _ = base
        self.pan_x = canvas_x - ox - (ix * s) * self.zoom
        self.pan_y = canvas_y - oy - (iy * s) * self.zoom

    def on_mousewheel(self, event):
        if event.delta == 0:
            return
        factor = ZOOM_STEP if event.delta > 0 else (1.0 / ZOOM_STEP)
        self.zoom_at(event.x, event.y, factor)
        self.redraw()

    def on_mousewheel_linux(self, event):
        if event.num == 4:
            factor = ZOOM_STEP
        elif event.num == 5:
            factor = 1.0 / ZOOM_STEP
        else:
            return
        self.zoom_at(event.x, event.y, factor)
        self.redraw()

    # ----------------- Editing -----------------
    def undo(self):
        """
        Undo removes ONLY the last point placed (never deletes a whole finished object).
        If there is no current object being drawn, it "re-opens" the last finished object
        and removes its last point.
        """
        if self.mode.get() == "obb":
            if self.obb_current:
                self.obb_current.pop()
            elif self.obbs:
                self.obb_current = self.obbs.pop()
                if self.obb_current:
                    self.obb_current.pop()
        else:
            if self.kpt_current:
                self.kpt_current.pop()
            elif self.kpt_sets:
                self.kpt_current = self.kpt_sets.pop()
                if self.kpt_current:
                    self.kpt_current.pop()

        self.redraw()

    def delete_last_object(self):
        """
        Delete removes the last OBJECT (box / keypoint-object) in the active mode.
        If you are currently drawing an object, it clears that current object first.
        """
        if self.mode.get() == "obb":
            if self.obb_current:
                self.obb_current = []
            elif self.obbs:
                self.obbs.pop()
        else:
            if self.kpt_current:
                self.kpt_current = []
            elif self.kpt_sets:
                self.kpt_sets.pop()

        self.redraw()

    def clear_current(self):
        self.obbs = []
        self.obb_current = []
        self.kpt_sets = []
        self.kpt_current = []
        self.guide_points = []
        self.guide_mode = False
        self._drag_point = None
        self.redraw()

    # ----------------- Export -----------------
    def save_labels(self):
        if self.current_image is None:
            return

        iw, ih = self.current_image.size

        if self.obb_current:
            messagebox.showinfo("Incomplete OBB", "You have an in-progress OBB (not 4 points). Finish it or Undo/Delete/Clear.")
            return
        if self.kpt_current:
            messagebox.showinfo(
                "Incomplete keypoints",
                f"You have in-progress keypoints (need {NUM_KEYPOINTS_PER_OBJECT} per object). Finish or Undo/Delete/Clear."
            )
            return

        # ---- Save OBB txt ----
        obb_path = self.obb_label_path()
        with open(obb_path, "w", encoding="utf-8") as f:
            for obb in self.obbs:
                if len(obb) != 4:
                    continue
                vals = [str(CLASS_ID)]
                for (x, y) in obb:
                    xn, yn = norm_xy(x, y, iw, ih)
                    vals.append(f"{xn:.16f}")
                    vals.append(f"{yn:.16f}")
                f.write(" ".join(vals) + "\n")

        # ---- Save keypoints txt ----
        kpt_path = self.kpt_label_path()
        with open(kpt_path, "w", encoding="utf-8") as f:
            for kps in self.kpt_sets:
                if len(kps) != NUM_KEYPOINTS_PER_OBJECT:
                    continue

                xs = [p[0] for p in kps]
                ys = [p[1] for p in kps]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                bw = max(1e-9, (x_max - x_min))
                bh = max(1e-9, (y_max - y_min))
                cx = x_min + bw / 2.0
                cy = y_min + bh / 2.0

                cxn, cyn = cx / iw, cy / ih
                bwn, bhn = bw / iw, bh / ih

                vals = [str(CLASS_ID),
                        f"{cxn:.16f}", f"{cyn:.16f}",
                        f"{bwn:.16f}", f"{bhn:.16f}"]

                for (x, y) in kps:
                    xn, yn = norm_xy(x, y, iw, ih)
                    vals.append(f"{xn:.16f}")
                    vals.append(f"{yn:.16f}")
                    vals.append(str(KEYPOINT_VISIBILITY))

                f.write(" ".join(vals) + "\n")

        self.redraw()

    # ----------------- Keys -----------------
    def on_key(self, e):
        k = e.keysym.lower()
        if k == "n":
            self.next_image()
        elif k == "p":
            self.prev_image()
        elif k == "s":
            self.save_labels()
        elif k == "u":
            self.undo()
        elif k == "d":
            self.delete_last_object()
        elif k == "c":
            self.clear_current()
        elif k == "o":
            self.mode.set("obb")
            self.redraw()
        elif k == "k":
            self.mode.set("kpt")
            self.redraw()
        elif k == "r":
            self.reset_view()
        elif k == "g":
            # Toggle guide mode; clears any partial guide points
            self.guide_mode = not self.guide_mode
            self.guide_points = []
            self.redraw()


def main():
    root = tk.Tk()
    app = AnnotatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()