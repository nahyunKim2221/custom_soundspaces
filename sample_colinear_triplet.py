# sample_collinear_triplets.py
import json, random, itertools, argparse
import numpy as np
from pathlib import Path

def load_points(txt_path):
    points = {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points[idx] = np.array([x, y, z])
    return points

def are_collinear(p1, p2, p3, plane="xz", tol=0.05):
    """
    세 점이 직선상에 있는지 확인.
    plane: 'xz' (수평면) | '3d' (전체 3D)
    tol: cross product magnitude 허용 오차
    """
    if plane == "xz":
        a = np.array([p1[0], p1[2]])
        b = np.array([p2[0], p2[2]])
        c = np.array([p3[0], p3[2]])
        ab = b - a
        ac = c - a
        cross = abs(ab[0]*ac[1] - ab[1]*ac[0])
        norm = np.linalg.norm(ab) * np.linalg.norm(ac) + 1e-9
    else:
        ab = p2 - p1
        ac = p3 - p1
        cross = np.linalg.norm(np.cross(ab, ac))
        norm = np.linalg.norm(ab) * np.linalg.norm(ac) + 1e-9

    return (cross / norm) < tol

def sample_collinear_triplets(
    jsonl_path, txt_path, output_path,
    plane="xz", tol=0.05,
    max_triplets=None, seed=42
):
    random.seed(seed)
    points = load_points(txt_path)

    results = []
    skipped_speakers = 0
    total_valid = 0

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            spk_id = entry["speaker"]
            listeners = list(dict.fromkeys(entry["listener"]))  # 중복 제거, 순서 유지

            if spk_id not in points:
                skipped_speakers += 1
                continue

            # listener 중 좌표가 있는 것만 사용
            valid_listeners = [l for l in listeners if l in points]

            if len(valid_listeners) < 3:
                continue

            # 모든 3-조합 검사
            collinear_triplets = []
            for trio in itertools.combinations(valid_listeners, 3):
                p1, p2, p3 = points[trio[0]], points[trio[1]], points[trio[2]]
                if are_collinear(p1, p2, p3, plane=plane, tol=tol):
                    collinear_triplets.append(list(trio))

            if not collinear_triplets:
                continue

            # max_triplets가 설정된 경우 랜덤 샘플링
            if max_triplets is not None and len(collinear_triplets) > max_triplets:
                collinear_triplets = random.sample(collinear_triplets, max_triplets)

            total_valid += len(collinear_triplets)

            spk_coords = points[spk_id].tolist()

            for trio in collinear_triplets:
                results.append({
                    "speaker": spk_id,
                    "speaker_coords": spk_coords,
                    "listeners": trio,
                    "listener_coords": [points[l].tolist() for l in trio],
                })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"완료: {len(results)}개 triplet 저장 ({total_valid} valid, {skipped_speakers} speaker 좌표 없음)")
    print(f"저장 경로: {output_path}")

# ── 데이터로더 헬퍼 ─────────────────────────────────────────────
class CollinearTripletDataset:
    """
    간단한 데이터로더용 Dataset 클래스 (PyTorch 없이도 동작).
    torch.utils.data.Dataset을 상속하려면 import 후 교체하면 됨.
    """
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "speaker":          s["speaker"],
            "speaker_coords":   np.array(s["speaker_coords"],   dtype=np.float32),
            "listeners":        s["listeners"],
            "listener_coords":  np.array(s["listener_coords"],  dtype=np.float32),  # (3, 3)
        }

# ── CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",   required=True,  help="speaker-listener JSONL 파일")
    parser.add_argument("--points",  required=True,  help="points_dense0.1.txt 경로")
    parser.add_argument("--output",  default="collinear_triplets.jsonl")
    parser.add_argument("--plane",   default="xz", choices=["xz", "3d"],
                        help="collinearity 체크 평면 (default: xz 수평면)")
    parser.add_argument("--tol",     type=float, default=0.05,
                        help="sin(각도) 허용 오차 (default: 0.05 ≈ 2.9°)")
    parser.add_argument("--max_triplets", type=int, default=None,
                        help="speaker당 최대 triplet 수 (default: 전부)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    sample_collinear_triplets(
        jsonl_path=args.jsonl,
        txt_path=args.points,
        output_path=args.output,
        plane=args.plane,
        tol=args.tol,
        max_triplets=args.max_triplets,
        seed=args.seed,
    )
