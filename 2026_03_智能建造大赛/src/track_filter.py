#!/usr/bin/env python3
"""
track_filter.py · 轻量 IoU 时序一致性滤波(ByteTrack-lite)

为什么需要(直击赛题痛点):水面反光/波纹会造成**单帧闪现的假框**。
要求"连续 ≥ min_hits 帧命中同一目标才确认显示",可几乎零算力地:
  ① 滤掉反光闪点(只出现 1-2 帧的假阳)；② 用 EMA 平滑框,demo 里框不再抖。
对救援场景:确认的目标即使偶尔漏检也会"滑行(coast)"保留 max_age 帧,避免目标闪烁丢失。

纯 Python 实现(无第三方依赖),可直接 `python track_filter.py` 跑自测。
在实时管线里:每帧把检测 [[x1,y1,x2,y2,score,cls],...] 喂给 update(),取 confirmed 框去画。
"""
from dataclasses import dataclass, field


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    tid: int
    box: list           # [x1,y1,x2,y2]
    cls: int
    score: float
    hits: int = 1       # 连续命中帧数
    misses: int = 0     # 连续未命中帧数
    age: int = 0
    confirmed: bool = False


@dataclass
class TrackFilter:
    """min_hits: 连续命中多少帧才确认(滤闪点);max_age: 确认后最多滑行多少帧才删;
    iou_thr: 帧间关联阈值;ema: 框平滑系数(0=不平滑,越大越平滑)。"""
    min_hits: int = 3
    max_age: int = 8
    iou_thr: float = 0.3
    ema: float = 0.5
    _tracks: list = field(default_factory=list)
    _next_id: int = 0

    def update(self, dets):
        """dets: [[x1,y1,x2,y2,score,cls], ...] -> 返回当前应显示的已确认 Track 列表。"""
        for t in self._tracks:
            t.age += 1
        used = [False] * len(dets)
        # 贪心 IoU 关联(同类优先)
        for t in self._tracks:
            best_j, best = -1, self.iou_thr
            for j, d in enumerate(dets):
                if used[j] or int(d[5]) != t.cls:
                    continue
                v = iou(t.box, d[:4])
                if v >= best:
                    best, best_j = v, j
            if best_j >= 0:
                d = dets[best_j]; used[best_j] = True
                a = self.ema
                t.box = [a*t.box[i] + (1-a)*d[i] for i in range(4)]  # EMA 平滑
                t.score, t.hits, t.misses = d[4], t.hits + 1, 0
                if t.hits >= self.min_hits:
                    t.confirmed = True
            else:
                t.misses += 1
                t.hits = 0
        # 未关联的检测 → 新建试探轨迹
        for j, d in enumerate(dets):
            if not used[j]:
                self._tracks.append(Track(self._next_id, list(d[:4]), int(d[5]), d[4]))
                self._next_id += 1
        # 删除久未命中的
        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]
        # 显示集 = 已确认 且 仍在滑行窗口内(misses<=max_age)。
        # 漏检帧用上一帧 EMA 平滑框继续画(coast),避免目标闪烁丢失——
        # 这正是 docstring 承诺的"确认目标偶尔漏检也滑行保留 max_age 帧"。
        # (misses>max_age 的已在上一行被删除,故等价于"未被删除的已确认轨迹"。)
        return [t for t in self._tracks if t.confirmed]


def _selftest():
    import sys
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    # 场景1:持续目标,min_hits=3 → 第3帧确认
    tf = TrackFilter(min_hits=3, max_age=5, iou_thr=0.3)
    box = [100, 100, 140, 140, 0.9, 0]
    out1 = tf.update([box]); out2 = tf.update([[101, 101, 141, 141, 0.9, 0]])
    check(len(out1) == 0 and len(out2) == 0, "持续目标前2帧未确认(试探中)")
    out3 = tf.update([[102, 102, 142, 142, 0.9, 0]])
    check(len(out3) == 1, "持续目标第3帧确认显示")

    # 场景2:单帧反光闪点 → 永不确认(被滤掉)
    tf2 = TrackFilter(min_hits=3, max_age=5, iou_thr=0.3)
    blip = tf2.update([[500, 500, 520, 520, 0.8, 0]])     # 闪一帧
    empty1 = tf2.update([])                                # 消失
    empty2 = tf2.update([])
    check(len(blip) == 0 and len(empty1) == 0 and len(empty2) == 0, "单帧反光闪点被滤除(从不确认)")

    # 场景3:确认后漏检仍滑行保留并继续显示(coast),超过 max_age 才彻底消失
    tf3 = TrackFilter(min_hits=2, max_age=3, iou_thr=0.3)
    tf3.update([[10, 10, 30, 30, 0.9, 1]]); c = tf3.update([[11, 11, 31, 31, 0.9, 1]])
    check(len(c) == 1, "min_hits=2 第2帧确认")
    tid = c[0].tid
    miss = tf3.update([])  # 漏检1帧
    check(any(t.tid == tid for t in tf3._tracks), "确认目标漏检1帧仍在(滑行保留)")
    # 关键:滑行帧必须仍在【显示集】里(用上一帧框继续画),否则就是 docstring 要消除的闪烁
    check(any(t.tid == tid for t in miss), "滑行帧仍被显示(coast 画上一帧框,不闪烁)")
    tf3.update([]); tf3.update([])          # 累计 misses=3 (== max_age),仍应保留
    still = tf3.update([])                   # misses=4 (> max_age),应被删除并不再显示
    check(all(t.tid != tid for t in still) and all(t.tid != tid for t in tf3._tracks),
          "滑行超过 max_age 帧后目标被删除(不再显示)")

    print("\n" + ("✅ track_filter 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
