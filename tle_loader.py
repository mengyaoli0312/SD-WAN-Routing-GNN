# tle_loader.py
from sgp4.api import Satrec, jday
import numpy as np
from astropy.time import Time
import re

class Satellite:
    def __init__(self, satnum, line1, line2):
        self.satnum = satnum
        self.line1 = line1
        self.line2 = line2
        self.satrec = Satrec.twoline2rv(line1, line2)

    def position_eci(self, t: Time) -> np.ndarray:
        if not isinstance(t, Time):
            t = Time(t, scale="utc")

        jd = int(t.utc.jd)
        fr = t.utc.jd % 1

        e, r, v = self.satrec.sgp4(jd, fr)

        if e != 0:
            print(f"[WARN] SGP4 error {e} for sat {self.satnum} at {t.isot}, using r anyway")

        return np.array(r, dtype=float)


def load_tle_file(path: str, max_sats: int | None = None):
    """
    支持两种格式：
    1) 传统 TLE:   line1, line2
    2) 含 line0:  0 NAME, line1, line2

    自动检测并跳过 line 0
    """
    sats = []

    with open(path, "r", encoding="utf-8") as f:
        # 清理空行
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    sat_counter = 0

    while i < len(raw_lines):

        # --- 情况 1: line0 存在 ---
        if raw_lines[i].startswith("0 "):
            # 下一行必须是 line1
            if i + 2 < len(raw_lines) and raw_lines[i+1].startswith("1 ") and raw_lines[i+2].startswith("2 "):
                line1 = raw_lines[i+1]
                line2 = raw_lines[i+2]
                i += 3
            else:
                print(f"[WARN] Bad TLE block starting at line: {raw_lines[i]}")
                i += 1
                continue

        # --- 情况 2: 无 line0, 直接 line1/line2 ---
        elif raw_lines[i].startswith("1 "):
            if i + 1 < len(raw_lines) and raw_lines[i+1].startswith("2 "):
                line1 = raw_lines[i]
                line2 = raw_lines[i+1]
                i += 2
            else:
                print(f"[WARN] Incomplete TLE pair at: {raw_lines[i]}")
                i += 1
                continue

        else:
            print(f"[WARN] Skip unknown line: {raw_lines[i]}")
            i += 1
            continue

        # 成功匹配到一对 TLE
        try:
            sat_counter += 1
            sat = Satellite(sat_counter, line1, line2)
            sats.append(sat)
        except Exception as e:
            print(f"[WARN] Failed to create Satellite: {e}")

        if max_sats is not None and len(sats) >= max_sats:
            break

    print(f"[INFO] Loaded {len(sats)} satellites from {path}")
    return sats
