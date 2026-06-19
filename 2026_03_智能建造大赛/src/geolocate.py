#!/usr/bin/env python3
"""
geolocate.py · 检测框中心 → 目标 GPS 经纬度(针孔模型 + 海平面求交)

赛题创新点①(国三→国一分水岭):不止画框,还把"落水人员/浮标"的像素位置反算成
**可航行的经纬度坐标**——这正是命题方(大江智造,大疆系)"发现落水者并给救援航点"的业务语言。

原理:
  无人机在海面上方高度 h,机载相机(此版假设云台增稳、可设前倾角)。给定检测框中心像素 (u,v)、
  相机内参、无人机 GPS+高度+航向+云台前倾角,做:
    像素 → 相机系射线 → 世界系射线(ENU)→ 与海平面 z=0 求交 → 局部 East/North 偏移 → 经纬度。

坐标约定(务必对齐真机):
  世界系 ENU(x=东, y=北, z=上),海平面 z=0,无人机位于 z=h。
  航向 yaw:正北为 0,顺时针为正(航空惯例);图像上方=机头前向。
  云台前倾 gimbal_tilt:0=正下方(nadir),正值=光轴从正下方向"前向"抬起。
  像素 (u,v):原点左上,u 向右,v 向下。

⚠️ 诚实声明:精度取决于高度/姿态/内参标定,实战误差在数十米量级(用已知 GPS 浮标做真值校验并报误差区间,
   别吹米级)。本版假设云台增稳忽略机体 roll/pitch;真机若用机体直挂相机需补 roll/pitch(见 TODO)。
   __main__ 自测验证的是 nadir/前倾/航向/经纬度换算的几何正确性(纯数学,可在任意机器跑)。
"""
import math
from dataclasses import dataclass

EARTH_M_PER_DEG = 111320.0  # 每纬度米数(近似)


# ----------------------------- 内参 / 位姿 -----------------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width, height, hfov_deg):
        """由水平视场角与分辨率估算内参(无标定时的近似)。"""
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        fy = fx  # 方形像素近似
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0)


@dataclass
class Pose:
    lat: float            # 无人机纬度(度)
    lon: float            # 无人机经度(度)
    alt_agl: float        # 距海面高度(米)
    yaw_deg: float = 0.0          # 航向:正北0,顺时针为正
    gimbal_tilt_deg: float = 0.0  # 云台前倾:0=正下方,正值向前抬


# ----------------------------- 3x3 线代(纯 Python,免依赖) -----------------------------
def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def _Rx(t):
    c, s = math.cos(t), math.sin(t)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _Rz_cw(psi):
    """绕世界 Up 轴顺时针(从上往下看)旋转 psi。"""
    c, s = math.cos(psi), math.sin(psi)
    return [[c, s, 0], [-s, c, 0], [0, 0, 1]]


# nadir、yaw=0 时的 相机系→世界系:x_c(右)→东, y_c(下)→南, z_c(光轴/前)→下
_R0 = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]


def _cam_to_world(pose: Pose):
    R_tilt = _Rx(math.radians(pose.gimbal_tilt_deg))   # 绕东轴前倾(把光轴从下抬向北)
    R_yaw = _Rz_cw(math.radians(pose.yaw_deg))         # 航向
    return _matmul(R_yaw, _matmul(R_tilt, _R0))


# ----------------------------- 核心:像素 → 经纬度 -----------------------------
def geolocate(u, v, intr: Intrinsics, pose: Pose):
    """返回 dict(lat, lon, east_m, north_m, ground_dist_m, slant_dist_m);射线打不到海面返回 None。"""
    ray_cam = [(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0]
    dx, dy, dz = _matvec(_cam_to_world(pose), ray_cam)   # 世界系方向(东,北,上)
    if dz >= -1e-9:                                       # 不朝下 → 打不到海面(指向地平线以上)
        return None
    t = pose.alt_agl / (-dz)                              # 从 (0,0,h) 沿射线到 z=0
    east, north = t * dx, t * dy
    dlat = north / EARTH_M_PER_DEG
    dlon = east / (EARTH_M_PER_DEG * math.cos(math.radians(pose.lat)))
    ground = math.hypot(east, north)
    return {
        "lat": pose.lat + dlat,
        "lon": pose.lon + dlon,
        "east_m": east,
        "north_m": north,
        "ground_dist_m": ground,
        "slant_dist_m": math.hypot(ground, pose.alt_agl),
    }


def geolocate_box(box_xyxy, intr: Intrinsics, pose: Pose):
    """检测框 [x1,y1,x2,y2] → 取底边中点(目标与水面接触处更接近真实落点)做定位。"""
    x1, y1, x2, y2 = box_xyxy
    return geolocate((x1 + x2) / 2.0, y2, intr, pose)


# ----------------------------- 真机:从 MAVLink 读位姿(可选) -----------------------------
def read_pose_mavlink(conn_str="udp:127.0.0.1:14550", timeout=5.0):
    """从飞控读 GLOBAL_POSITION_INT(经纬高) + ATTITUDE(航向)。需 pip install pymavlink。"""
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(conn_str)
    m.wait_heartbeat(timeout=timeout)
    gpi = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
    att = m.recv_match(type="ATTITUDE", blocking=True, timeout=timeout)
    return Pose(
        lat=gpi.lat / 1e7, lon=gpi.lon / 1e7,
        alt_agl=gpi.relative_alt / 1000.0,
        yaw_deg=math.degrees(att.yaw) % 360.0,
        gimbal_tilt_deg=0.0,  # TODO: 接云台 MOUNT_ORIENTATION 读真实倾角
    )


# ----------------------------- 自测(纯数学,任意机器可跑) -----------------------------
def _selftest():
    import sys
    H, W, h = 1080, 1920, 100.0           # 100m 高度
    intr = Intrinsics.from_fov(W, H, hfov_deg=84.0)
    base = dict(lat=31.20, lon=121.50, alt_agl=h)
    ok = True

    def approx(a, b, tol=0.5, msg=""):
        nonlocal ok
        if abs(a - b) > tol:
            ok = False; print(f"  ❌ {msg}: 得 {a:.3f} 期望 {b:.3f}")
        else:
            print(f"  ✅ {msg}: {a:.3f} ≈ {b:.3f}")

    # A. nadir 正下方,中心像素 → 偏移≈0
    r = geolocate(intr.cx, intr.cy, intr, Pose(**base))
    approx(r["ground_dist_m"], 0.0, 0.01, "A 中心像素落点在正下方")

    # B. nadir, 中心右移 0.1*fx 像素 → 东偏 ≈ h*0.1
    r = geolocate(intr.cx + 0.1 * intr.fx, intr.cy, intr, Pose(**base))
    approx(r["east_m"], h * 0.1, 0.05, "B 右移→东偏 h*0.1")
    approx(r["north_m"], 0.0, 0.05, "B 右移→北偏0")

    # C. nadir, 中心下移 0.1*fy → 南偏(north 负)≈ -h*0.1(图像下=南)
    r = geolocate(intr.cx, intr.cy + 0.1 * intr.fy, intr, Pose(**base))
    approx(r["north_m"], -h * 0.1, 0.05, "C 下移→南偏 -h*0.1")

    # D. 经纬度换算:北偏 EARTH_M_PER_DEG 米 → 纬度+1.0(构造北偏:中心上移)
    r = geolocate(intr.cx, intr.cy - (EARTH_M_PER_DEG / h) * intr.fy, intr, Pose(**base))
    approx(r["lat"] - base["lat"], 1.0, 1e-3, "D 北偏111320m→纬度+1.0")

    # E. 航向东(yaw=90),中心右移 → 应南偏(东向右=南),东偏≈0
    r = geolocate(intr.cx + 0.1 * intr.fx, intr.cy, intr, Pose(**base, yaw_deg=90))
    approx(r["north_m"], -h * 0.1, 0.05, "E yaw=90 右移→南偏")
    approx(r["east_m"], 0.0, 0.05, "E yaw=90 右移→东偏0")

    # F. 云台前倾 30°,中心像素 → 正前(北)偏 ≈ h*tan(30)
    r = geolocate(intr.cx, intr.cy, intr, Pose(**base, gimbal_tilt_deg=30))
    approx(r["north_m"], h * math.tan(math.radians(30)), 0.1, "F 前倾30°中心→北偏 h*tan30")

    # 演示一条:框→GPS
    demo = geolocate_box([900, 500, 960, 560], intr, Pose(lat=31.20, lon=121.50, alt_agl=80, yaw_deg=45, gimbal_tilt_deg=20))
    print(f"\n  示例 框[900,500,960,560]@80m,航向45°,前倾20° → "
          f"目标≈({demo['lat']:.6f},{demo['lon']:.6f}), 距本艇 {demo['ground_dist_m']:.1f}m")

    print("\n" + ("✅ 全部几何自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
