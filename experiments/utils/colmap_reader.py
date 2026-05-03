"""
COLMAP binary-format reader.
Adapted from mip-splatting/scene/colmap_loader.py with additions:
  - read_points3D_binary_with_ids: preserves point3D_id keys
  - build_sparse_depth: projects 3D points into each image frame
  - interpolate_depth: dense interpolation from sparse depth samples
"""

import struct
import collections
import numpy as np
from scipy.interpolate import griddata

# ── namedtuples ───────────────────────────────────────────────────────────────
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error"])

CAMERA_MODELS = {
    CameraModel(0,  "SIMPLE_PINHOLE",      3),
    CameraModel(1,  "PINHOLE",             4),
    CameraModel(2,  "SIMPLE_RADIAL",       4),
    CameraModel(3,  "RADIAL",              5),
    CameraModel(4,  "OPENCV",              8),
    CameraModel(5,  "OPENCV_FISHEYE",      8),
    CameraModel(6,  "FULL_OPENCV",        12),
    CameraModel(7,  "FOV",                 5),
    CameraModel(8,  "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9,  "RADIAL_FISHEYE",      5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}


# ── helpers ───────────────────────────────────────────────────────────────────
def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def qvec2rotmat(qvec):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix (world→camera)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y],
    ])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


# ── binary readers ────────────────────────────────────────────────────────────
def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(fid, 24, "iiQQ")
            cam_id, model_id, width, height = props
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = Camera(
                id=cam_id, model=model_name,
                width=width, height=height,
                params=np.array(params))
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            while True:
                ch = fid.read(1)
                if ch == b"\x00":
                    break
                name += ch.decode("utf-8")
            n_pts = _read_next_bytes(fid, 8, "Q")[0]
            raw = _read_next_bytes(fid, 24 * n_pts, "ddq" * n_pts)
            xys = np.column_stack([
                list(map(float, raw[0::3])),
                list(map(float, raw[1::3])),
            ])
            point3D_ids = np.array(list(map(int, raw[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_binary_with_ids(path):
    """Returns dict {point3D_id: Point3D} preserving IDs."""
    points = {}
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            raw = _read_next_bytes(fid, 43, "QdddBBBd")
            pid = raw[0]
            xyz   = np.array(raw[1:4])
            rgb   = np.array(raw[4:7], dtype=np.uint8)
            error = float(raw[7])
            track_len = _read_next_bytes(fid, 8, "Q")[0]
            # skip track data (image_id, point2D_idx pairs)
            fid.read(8 * track_len)
            points[pid] = Point3D(id=pid, xyz=xyz, rgb=rgb, error=error)
    return points


# ── camera intrinsic helpers ──────────────────────────────────────────────────
def camera_params_to_K(camera: Camera):
    """Return 3×3 intrinsic matrix for PINHOLE / SIMPLE_PINHOLE cameras."""
    p = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    elif camera.model == "PINHOLE":
        fx = p[0]; fy = p[1]; cx = p[2]; cy = p[3]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def scale_K(K: np.ndarray, orig_wh, new_wh):
    """Scale intrinsic matrix from orig_wh=(W,H) to new_wh=(W,H)."""
    sx = new_wh[0] / orig_wh[0]
    sy = new_wh[1] / orig_wh[1]
    K_new = K.copy()
    K_new[0, 0] *= sx; K_new[0, 2] *= sx
    K_new[1, 1] *= sy; K_new[1, 2] *= sy
    return K_new


# ── sparse depth ──────────────────────────────────────────────────────────────
def build_sparse_depth(
    colmap_image: Image,
    camera: Camera,
    points3D: dict,
    target_wh=None,
):
    """
    Project visible 3D points into the image to build a sparse depth map.

    Args:
        colmap_image: Image namedtuple with qvec, tvec, xys, point3D_ids
        camera: Camera namedtuple
        points3D: dict {id: Point3D}
        target_wh: (W, H) of the target map; if None, uses camera's native size

    Returns:
        u, v, d  – pixel coords and metric depths of valid sparse points
    """
    R = qvec2rotmat(colmap_image.qvec)   # world→camera
    t = colmap_image.tvec

    K = camera_params_to_K(camera)
    orig_wh = (camera.width, camera.height)
    if target_wh is not None:
        K = scale_K(K, orig_wh, target_wh)
        W, H = target_wh
    else:
        W, H = orig_wh

    us, vs, ds = [], [], []
    for (x2d, y2d), pid in zip(colmap_image.xys, colmap_image.point3D_ids):
        if pid == -1 or pid not in points3D:
            continue
        P_world = points3D[pid].xyz
        P_cam   = R @ P_world + t
        d = P_cam[2]
        if d <= 0:
            continue
        # Scale 2D keypoint from native resolution to target resolution
        if target_wh is not None:
            sx = target_wh[0] / orig_wh[0]
            sy = target_wh[1] / orig_wh[1]
            x2d_s = x2d * sx
            y2d_s = y2d * sy
        else:
            x2d_s, y2d_s = x2d, y2d
        if 0 <= x2d_s < W and 0 <= y2d_s < H:
            us.append(x2d_s)
            vs.append(y2d_s)
            ds.append(d)

    return np.array(us), np.array(vs), np.array(ds)


def interpolate_depth(us, vs, ds, W, H, method="linear", fill_value=None):
    """
    Interpolate sparse (u,v,d) samples to a dense W×H depth map.

    Missing regions are filled with the median depth or fill_value.
    """
    if len(ds) == 0:
        median_d = 1.0
        return np.full((H, W), median_d, dtype=np.float32)

    # Dense grid
    grid_u, grid_v = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    dense = griddata(
        points=np.stack([us, vs], axis=1),
        values=ds,
        xi=(grid_u, grid_v),
        method=method,
    )
    # Fill NaN with nearest-neighbor
    mask = np.isnan(dense)
    if mask.any():
        dense_nn = griddata(
            points=np.stack([us, vs], axis=1),
            values=ds,
            xi=(grid_u, grid_v),
            method="nearest",
        )
        dense[mask] = dense_nn[mask]
    return dense.astype(np.float32)
