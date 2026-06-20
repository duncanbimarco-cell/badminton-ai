import cv2
import mediapipe as mp
import os
import math
import numpy as np
import csv
import json
import argparse
import traceback

# ==================== 配置 ====================
IS_LEFT_HANDED = True

# MediaPipe Pose 关键点索引
# 左臂: 11(肩) 13(肘) 15(腕)   右臂: 12(肩) 14(肘) 16(腕)
if IS_LEFT_HANDED:
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 11, 13, 15
    SIDE_LABEL = "左手"
else:
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 12, 14, 16
    SIDE_LABEL = "右手"

if os.environ.get("BADMINTON_LEFT_HANDED") == "1":
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 11, 13, 15
    SIDE_LABEL = "左手"
elif os.environ.get("BADMINTON_LEFT_HANDED") == "0":
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 12, 14, 16
    SIDE_LABEL = "右手"

ARM_GREY  = (128, 128, 128)
ARM_GREEN = (0, 255, 0)

SWING_ANGLE_MIN = 80
SWING_ANGLE_MAX = 170

DEBUG = True

MOVING_AVG_WINDOW = 5
IMPACT_SPEED_THRESHOLD = 10.0
BACKSWING_ANGLE_MIN = 120
BACKSWING_ANGLE_MAX = 160
OUTPUT_CSV = "swing_data.csv"
OUTPUT_JSON = "swing_report.json"
# ================================================


def calc_elbow_angle(shoulder, elbow, wrist):
    v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    v2 = (wrist[0] - elbow[0],     wrist[1] - elbow[1])
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    angle = abs(math.degrees(a2 - a1))
    return 360 - angle if angle > 180 else angle


def draw_arm_skeleton(frame, sx, sy, ex, ey, wx, wy,
                      c_shoulder, c_elbow, c_wrist, color, thickness=3, min_conf=0.3):
    for name, c in [("肩", c_shoulder), ("肘", c_elbow), ("腕", c_wrist)]:
        if c < min_conf:
            print(f"  [Skip] {SIDE_LABEL}{name} 可见度={c:.2f} < {min_conf}")

    if c_shoulder >= min_conf and c_elbow >= min_conf:
        if DEBUG:
            print(f"  [draw] 肩→肘  line ({sx},{sy}) → ({ex},{ey})  color={color}")
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness)
        cv2.circle(frame, (sx, sy), 6, color, -1)

    if c_elbow >= min_conf and c_wrist >= min_conf:
        if DEBUG:
            print(f"  [draw] 肘→腕  line ({ex},{ey}) → ({wx},{wy})  color={color}")
        cv2.line(frame, (ex, ey), (wx, wy), color, thickness)
        cv2.circle(frame, (ex, ey), 6, color, -1)
        cv2.circle(frame, (wx, wy), 6, color, -1)


# ==================== 离线分析函数 ====================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle


def moving_average(data, window):
    if len(data) < window:
        return np.array(data, dtype=np.float64)
    kernel = np.ones(window) / window
    smoothed_valid = np.convolve(data, kernel, mode='valid')
    pad_left = window // 2
    pad_right = window - pad_left - 1
    return np.pad(smoothed_valid, (pad_left, pad_right), mode='edge')


def detect_impact_frame(frames_data, smoothed_wy, window=MOVING_AVG_WINDOW):
    if len(smoothed_wy) < 2:
        return 0, 0.0
    delta_y = np.diff(smoothed_wy)
    margin = window // 2
    lo, hi = margin, max(len(delta_y) - margin, margin + 1)
    if lo >= hi:
        lo, hi = 0, len(delta_y)
    i_max = int(lo + np.argmax(delta_y[lo:hi]))
    impact_idx = i_max + 1
    impact_velocity = float(delta_y[i_max])
    return impact_idx, impact_velocity


def classify_swing_phases(frames_data, impact_idx):
    n = len(frames_data)
    phases = ['preparation'] * n
    for i, fd in enumerate(frames_data):
        angle = fd['elbow_angle']
        if i == impact_idx:
            phases[i] = 'impact'
        elif i > impact_idx:
            phases[i] = 'follow_through'
        elif BACKSWING_ANGLE_MIN <= angle <= BACKSWING_ANGLE_MAX:
            phases[i] = 'backswing'
        else:
            phases[i] = 'preparation'
    return phases


def generate_report(frames_data, phases, impact_idx, impact_vel):
    backswing_angles = [
        fd['elbow_angle'] for fd, p in zip(frames_data, phases)
        if p == 'backswing'
    ]
    max_backswing = max(backswing_angles) if backswing_angles else 0
    backswing_verdict = "优秀" if max_backswing > 150 else "建议增加挥拍幅度"
    impact_verdict = "高" if abs(impact_vel) > IMPACT_SPEED_THRESHOLD else "发力不足"
    follow_count = sum(1 for p in phases if p == 'follow_through')
    motion_verdict = "完整" if follow_count > 20 else "动作过早中断"

    return {
        "蓄力评估": backswing_verdict,
        "击球爆发力": impact_verdict,
        "动作连贯性": motion_verdict,
        "_detail": {
            "max_backswing_angle": round(max_backswing, 1),
            "impact_velocity_px_per_frame": round(impact_vel, 2),
            "impact_frame": impact_idx,
            "follow_through_frames": follow_count,
            "total_frames": len(frames_data),
        }
    }


def save_csv(frames_data, phases, smoothed_wy, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame', 'shoulder_x', 'shoulder_y',
            'elbow_x', 'elbow_y', 'wrist_x', 'wrist_y',
            'elbow_angle', 'phase', 'smoothed_wrist_y'
        ])
        for i, fd in enumerate(frames_data):
            writer.writerow([
                fd['frame'],
                round(fd['shoulder_x'], 2), round(fd['shoulder_y'], 2),
                round(fd['elbow_x'], 2),    round(fd['elbow_y'], 2),
                round(fd['wrist_x'], 2),    round(fd['wrist_y'], 2),
                round(fd['elbow_angle'], 2),
                phases[i],
                round(float(smoothed_wy[i]), 2),
            ])


def main():
    parser = argparse.ArgumentParser(description="🏸 羽毛球 AI 挥拍分析 (MediaPipe)")
    parser.add_argument("--input",  default="test_video.mp4")
    parser.add_argument("--output", default="output_processed.mp4")
    parser.add_argument("--csv",    default="swing_data.csv")
    parser.add_argument("--json",   default="swing_report.json")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    global OUTPUT_CSV, OUTPUT_JSON
    OUTPUT_CSV = args.csv
    OUTPUT_JSON = args.json

    print(f"持拍手: {SIDE_LABEL}  |  挥拍区间: {SWING_ANGLE_MIN}° – {SWING_ANGLE_MAX}°")
    print(f"模式: {'Headless' if args.headless else 'GUI 实时显示'}")
    print(f"输入: {args.input}")

    test_file = args.input
    if not os.path.exists(test_file):
        print(f"错误: 找不到 {test_file}")
        return

    # 初始化 MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    cap = cv2.VideoCapture(test_file)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {test_file}")
        pose.close()
        return

    frame_count = 0
    frame_shape_printed = False
    frames_data = []

    # Headless 模式：创建视频写入器
    out_writer = None
    if args.headless:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 尝试多种编码器
        codec_ok = False
        for codec in ['avc1', 'mp4v', 'XVID', 'MJPG']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
                if out_writer.isOpened():
                    print(f"输出视频: {args.output}  ({w}x{h}, {fps:.1f}fps, {codec})")
                    codec_ok = True
                    break
                out_writer.release()
                out_writer = None
            except Exception:
                out_writer = None
        if not codec_ok:
            print("警告: 无法创建视频输出文件，将跳过视频保存")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        h_img, w_img = frame.shape[:2]

        if not frame_shape_printed:
            print(f"Frame shape: {frame.shape}  (H={h_img}, W={w_img})")
            frame_shape_printed = True

        # MediaPipe 推理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks is None:
            if frame_count % 30 == 0:
                print(f"[Frame {frame_count}] Warning: No pose detected.")
        else:
            lm = results.pose_landmarks.landmark

            sx = lm[SHOULDER_IDX].x * w_img
            sy = lm[SHOULDER_IDX].y * h_img
            ex = lm[ELBOW_IDX].x   * w_img
            ey = lm[ELBOW_IDX].y   * h_img
            wx = lm[WRIST_IDX].x   * w_img
            wy = lm[WRIST_IDX].y   * h_img

            sc = lm[SHOULDER_IDX].visibility
            ec = lm[ELBOW_IDX].visibility
            wc = lm[WRIST_IDX].visibility

            if DEBUG:
                print(f"[Frame {frame_count}] {SIDE_LABEL} — "
                      f"肩:({sx:.1f},{sy:.1f}) v={sc:.2f}  "
                      f"肘:({ex:.1f},{ey:.1f}) v={ec:.2f}  "
                      f"腕:({wx:.1f},{wy:.1f}) v={wc:.2f}")

            coords_ok = all((x != 0 or y != 0) for x, y in ((sx, sy), (ex, ey), (wx, wy)))
            if not coords_ok:
                print(f"[Frame {frame_count}] Warning: 含 (0,0) 无效坐标")
            elif sc > 0.3 and ec > 0.3 and wc > 0.3:
                angle = calc_elbow_angle((sx, sy), (ex, ey), (wx, wy))
                in_swing = SWING_ANGLE_MIN <= angle <= SWING_ANGLE_MAX
                color = ARM_GREEN if in_swing else ARM_GREY

                frames_data.append({
                    'frame': frame_count,
                    'shoulder_x': sx, 'shoulder_y': sy,
                    'elbow_x': ex, 'elbow_y': ey,
                    'wrist_x': wx, 'wrist_y': wy,
                    'elbow_angle': angle,
                })

                draw_arm_skeleton(frame,
                    int(sx), int(sy), int(ex), int(ey), int(wx), int(wy),
                    sc, ec, wc, color)

                cv2.putText(frame, f"{SIDE_LABEL}肘角: {angle:.1f}",
                            (int(ex) + 20, int(ey) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if in_swing else (200, 200, 200), 2)

                if in_swing:
                    cv2.putText(frame, "SWING!", (40, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        if args.headless and out_writer is not None:
            out_writer.write(frame)
        elif not args.headless:
            cv2.imshow("Badminton AI (MediaPipe)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    pose.close()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

    # ========== 离线分析管线 ==========
    print("\n" + "=" * 50)
    print(f"离线分析: 共采集 {len(frames_data)} 帧")

    if len(frames_data) < MOVING_AVG_WINDOW:
        print(f"有效帧数不足 {MOVING_AVG_WINDOW}，跳过分析")
    else:
        raw_wrist_y = [fd['wrist_y'] for fd in frames_data]
        smoothed_wy = moving_average(raw_wrist_y, MOVING_AVG_WINDOW)

        impact_idx, impact_vel = detect_impact_frame(frames_data, smoothed_wy)
        print(f"击球帧: #{impact_idx}  手腕下移速度: {impact_vel:.2f} px/frame")

        phases = classify_swing_phases(frames_data, impact_idx)
        report = generate_report(frames_data, phases, impact_idx, impact_vel)

        save_csv(frames_data, phases, smoothed_wy, OUTPUT_CSV)
        print(f"CSV: {OUTPUT_CSV}")

        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"JSON: {OUTPUT_JSON}")

        print("-" * 40)
        print(f"  蓄力评估:    {report['蓄力评估']}")
        print(f"  击球爆发力:  {report['击球爆发力']}")
        print(f"  动作连贯性:  {report['动作连贯性']}")
        print(f"  最大蓄力角度: {report['_detail']['max_backswing_angle']}°")
        print(f"  跟随帧数:     {report['_detail']['follow_through_frames']}")
        print("=" * 50)

    print("Done.")


if __name__ == "__main__":
    main()
