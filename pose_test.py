import cv2
from ultralytics import YOLO
import torch
import os
import math
import numpy as np
import csv
import json
import argparse

# ==================== 配置 ====================
IS_LEFT_HANDED = True   # True=左手持拍, False=右手持拍

# 环境变量可覆盖默认值（供 app.py 等外部调用者使用）
if os.environ.get("BADMINTON_LEFT_HANDED") == "1":
    IS_LEFT_HANDED = True
elif os.environ.get("BADMINTON_LEFT_HANDED") == "0":
    IS_LEFT_HANDED = False

# 关键点索引 — 注意: YOLOv8-pose 使用 COCO 格式（非 MediaPipe）
# 左臂: 5(肩) 7(肘) 9(腕)   右臂: 6(肩) 8(肘) 10(腕)
if IS_LEFT_HANDED:
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 5, 7, 9
    SIDE_LABEL = "左手"
else:
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 6, 8, 10
    SIDE_LABEL = "右手"

ARM_GREY  = (128, 128, 128)   # 默认灰色 (BGR)
ARM_GREEN = (0, 255, 0)       # 挥拍区间绿色 (BGR)

SWING_ANGLE_MIN = 80          # 挥拍角度下限
SWING_ANGLE_MAX = 170         # 挥拍角度上限

DEBUG = True                  # 开启终端诊断输出

# ---- 离线分析配置 ----
MOVING_AVG_WINDOW = 5          # 移动平均窗口（帧）
IMPACT_SPEED_THRESHOLD = 10.0  # 击球爆发力阈值（像素/帧）
BACKSWING_ANGLE_MIN = 120      # 蓄力阶段肘角下限
BACKSWING_ANGLE_MAX = 160      # 蓄力阶段肘角上限
OUTPUT_CSV = "swing_data.csv"
OUTPUT_JSON = "swing_report.json"
# ================================================


def calc_elbow_angle(shoulder, elbow, wrist):
    """用 atan2 计算肘部内角 (肩-肘-腕)"""
    # 向量: 肘→肩, 肘→腕
    v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    v2 = (wrist[0] - elbow[0],     wrist[1] - elbow[1])

    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    angle = abs(math.degrees(a2 - a1))
    return 360 - angle if angle > 180 else angle


def draw_arm_skeleton(frame, kpts, shoulder, elbow, wrist, color, thickness=3, min_conf=0.3):
    """只绘制持拍手一侧: 肩→肘→腕 + 关节点 — 所有坐标强制 int()"""
    c_shoulder = float(kpts[SHOULDER_IDX][2])
    c_elbow   = float(kpts[ELBOW_IDX][2])
    c_wrist   = float(kpts[WRIST_IDX][2])

    sx, sy = int(shoulder[0]), int(shoulder[1])
    ex, ey = int(elbow[0]),    int(elbow[1])
    wx, wy = int(wrist[0]),    int(wrist[1])

    # 逐个关键点置信度检查
    for name, c, xy in [("肩", c_shoulder, (sx, sy)),
                         ("肘", c_elbow,   (ex, ey)),
                         ("腕", c_wrist,   (wx, wy))]:
        if c < min_conf:
            print(f"  [Skip] {SIDE_LABEL}{name} 置信度={c:.2f} < {min_conf}，不绘制该点连线")

    # 肩 → 肘（两点都需可信）
    if c_shoulder >= min_conf and c_elbow >= min_conf:
        if DEBUG:
            print(f"  [draw] 肩→肘  line ({sx},{sy}) → ({ex},{ey})  color={color}")
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness)
        cv2.circle(frame, (sx, sy), 6, color, -1)

    # 肘 → 腕（两点都需可信）
    if c_elbow >= min_conf and c_wrist >= min_conf:
        if DEBUG:
            print(f"  [draw] 肘→腕  line ({ex},{ey}) → ({wx},{wy})  color={color}")
        cv2.line(frame, (ex, ey), (wx, wy), color, thickness)
        cv2.circle(frame, (ex, ey), 6, color, -1)
        cv2.circle(frame, (wx, wy), 6, color, -1)


# ==================== 离线分析函数 ====================

def calculate_angle(a, b, c):
    """计算三点之间的夹角（以 b 为顶点），numpy 实现"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def moving_average(data, window):
    """Boxcar 移动平均平滑，返回与输入等长的 np.ndarray

    用 'valid' 模式避免零填充边缘伪影，再用边缘值补齐长度。
    """
    if len(data) < window:
        return np.array(data, dtype=np.float64)
    kernel = np.ones(window) / window
    # valid 模式：不依赖零填充，结果长度 = len(data) - window + 1
    smoothed_valid = np.convolve(data, kernel, mode='valid')
    pad_left = window // 2
    pad_right = window - pad_left - 1
    # 用边缘值补齐，不引入人为速度尖峰
    return np.pad(smoothed_valid, (pad_left, pad_right), mode='edge')


def detect_impact_frame(frames_data, smoothed_wy, window=MOVING_AVG_WINDOW):
    """通过手腕 Y 坐标下降速度最大帧定位击球瞬间

    返回 (impact_idx, impact_velocity)
    impact_idx: 击球帧在 frames_data 中的索引
    impact_velocity: 该帧手腕Y向速度（像素/帧，正值=向下）
    """
    if len(smoothed_wy) < 2:
        return 0, 0.0

    delta_y = np.diff(smoothed_wy)          # 相邻帧 Y 差值
    # 排除边缘区域（移动平均的补齐段），只在有效区间搜索
    margin = window // 2
    lo, hi = margin, max(len(delta_y) - margin, margin + 1)
    if lo >= hi:
        lo, hi = 0, len(delta_y)

    i_max = int(lo + np.argmax(delta_y[lo:hi]))  # Y 增加最快 = 手腕向下最快
    impact_idx = i_max + 1                         # 速度在 i→i+1 之间，取 i+1 为击球帧
    impact_velocity = float(delta_y[i_max])
    return impact_idx, impact_velocity


def classify_swing_phases(frames_data, impact_idx):
    """为每一帧标注挥拍阶段

    返回等长 list[str]，取值: preparation | backswing | impact | follow_through
    """
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
    """根据挥拍数据生成评估报告"""
    # 蓄力评估：取 backswing 阶段最大肘角
    backswing_angles = [
        fd['elbow_angle'] for fd, p in zip(frames_data, phases)
        if p == 'backswing'
    ]
    max_backswing = max(backswing_angles) if backswing_angles else 0
    backswing_verdict = "优秀" if max_backswing > 150 else "建议增加挥拍幅度"

    # 击球爆发力
    impact_verdict = "高" if abs(impact_vel) > IMPACT_SPEED_THRESHOLD else "发力不足"

    # 动作连贯性：击球后 follow_through 帧数
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
    """将逐帧数据写入 CSV"""
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

# ====================================================


def main():
    parser = argparse.ArgumentParser(description="🏸 羽毛球 AI 挥拍分析")
    parser.add_argument("--input",  default="test_video.mp4",       help="输入视频路径")
    parser.add_argument("--output", default="output_processed.mp4", help="输出标注视频路径（仅 headless 模式）")
    parser.add_argument("--csv",    default="swing_data.csv",       help="CSV 输出路径")
    parser.add_argument("--json",   default="swing_report.json",    help="JSON 报告路径")
    parser.add_argument("--headless", action="store_true",          help="无 GUI 模式（Web/服务器）")
    args = parser.parse_args()

    # 将参数写入全局配置，供 save_csv 等处使用
    global OUTPUT_CSV, OUTPUT_JSON
    OUTPUT_CSV = args.csv
    OUTPUT_JSON = args.json

    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device_str}")
    print(f"持拍手: {SIDE_LABEL}  |  挥拍区间: {SWING_ANGLE_MIN}° – {SWING_ANGLE_MAX}°")
    print(f"调试模式: {'开' if DEBUG else '关'}")
    print(f"关键点索引 — 肩:{SHOULDER_IDX} 肘:{ELBOW_IDX} 腕:{WRIST_IDX}")
    print(f"模式: {'Headless (输出视频)' if args.headless else 'GUI 实时显示'}")
    print(f"输入视频: {args.input}")

    model = YOLO("yolov8n-pose.pt")
    test_file = args.input

    if not os.path.exists(test_file):
        print(f"错误: 找不到 {test_file}，请确保视频文件在当前目录下。")
        return

    cap = cv2.VideoCapture(test_file)
    frame_count = 0
    frame_shape_printed = False
    frames_data = []  # 离线分析数据容器

    # Headless 模式：创建视频写入器
    out_writer = None
    if args.headless:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"输出视频: {args.output}  ({w}x{h}, {fps:.1f} fps)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # 打印一次图像尺寸，验证坐标与画布匹配
        if not frame_shape_printed:
            print(f"Frame shape: {frame.shape}  (H={frame.shape[0]}, W={frame.shape[1]})")
            frame_shape_printed = True

        results = model(frame, device=device_str, verbose=False)
        kp = results[0].keypoints          # ultralytics Keypoints 对象

        # ========== 诊断输出 ==========
        if kp is None or kp.xy is None or kp.xy.shape[0] == 0 or kp.xy.shape[1] == 0:
            print(f"[Frame {frame_count}] Warning: No keypoints detected in this frame.")
        else:
            # 直接用 results[0].keypoints.xy[0][索引] 获取正确坐标
            xy   = results[0].keypoints.xy[0]    # shape (17, 2), 第一个人
            conf = results[0].keypoints.conf[0]  # shape (17,)

            sx, sy = float(xy[SHOULDER_IDX][0]), float(xy[SHOULDER_IDX][1])
            ex, ey = float(xy[ELBOW_IDX][0]),    float(xy[ELBOW_IDX][1])
            wx, wy = float(xy[WRIST_IDX][0]),    float(xy[WRIST_IDX][1])
            sc, ec, wc = float(conf[SHOULDER_IDX]), float(conf[ELBOW_IDX]), float(conf[WRIST_IDX])

            if DEBUG:
                print(f"[Frame {frame_count}] {SIDE_LABEL} — "
                      f"肩:({sx:.1f},{sy:.1f}) c={sc:.2f}  "
                      f"肘:({ex:.1f},{ey:.1f}) c={ec:.2f}  "
                      f"腕:({wx:.1f},{wy:.1f}) c={wc:.2f}")

            # 检查坐标是否有效（非 None 且非零）
            coords_ok = all(
                (x != 0 or y != 0)
                for x, y in ((sx, sy), (ex, ey), (wx, wy))
            )
            if not coords_ok:
                print(f"[Frame {frame_count}] Warning: {SIDE_LABEL}关键点含 (0,0) 无效坐标，跳过绘制")
            elif sc > 0.3 and ec > 0.3 and wc > 0.3:
                shoulder = (sx, sy)
                elbow    = (ex, ey)
                wrist    = (wx, wy)

                angle = calc_elbow_angle(shoulder, elbow, wrist)
                in_swing = SWING_ANGLE_MIN <= angle <= SWING_ANGLE_MAX
                color = ARM_GREEN if in_swing else ARM_GREY

                # 采集数据用于离线分析
                frames_data.append({
                    'frame': frame_count,
                    'shoulder_x': sx, 'shoulder_y': sy,
                    'elbow_x': ex, 'elbow_y': ey,
                    'wrist_x': wx, 'wrist_y': wy,
                    'elbow_angle': angle,
                })

                # 自定义绘制：只画持拍手骨骼
                draw_arm_skeleton(frame, kp.data[0], shoulder, elbow, wrist, color)

                # ---- 角度数值 ----
                cv2.putText(frame, f"{SIDE_LABEL}肘角: {angle:.1f}",
                            (int(ex) + 20, int(ey) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if in_swing else (200, 200, 200), 2)

                # ---- 挥拍状态 ----
                if in_swing:
                    cv2.putText(frame, "SWING!", (40, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        # ========== 输出（Headless 写视频 / GUI 双窗口） ==========
        if args.headless:
            # Headless：将标注帧写入输出视频
            out_writer.write(frame)
        else:
            cv2.imshow("Badminton AI (Custom)", frame)
            # 备用窗口：始终显示原始 YOLO 骨骼，验证 AI 是否检测到人
            debug_plot = results[0].plot()
            cv2.imshow("Debug Skeleton (YOLO raw)", debug_plot)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

    # ========== 离线分析管线 ==========
    print("\n" + "=" * 50)
    print("离线分析开始...")
    print(f"共采集 {len(frames_data)} 帧有效数据")

    if len(frames_data) < MOVING_AVG_WINDOW:
        print(f"警告: 有效帧数不足 {MOVING_AVG_WINDOW}，跳过分析")
    else:
        # 1. 提取手腕 Y 坐标序列并平滑
        raw_wrist_y = [fd['wrist_y'] for fd in frames_data]
        smoothed_wy = moving_average(raw_wrist_y, MOVING_AVG_WINDOW)

        # 2. 定位击球帧
        impact_idx, impact_vel = detect_impact_frame(frames_data, smoothed_wy)
        print(f"击球帧: #{impact_idx}  手腕下移速度: {impact_vel:.2f} px/frame")

        # 3. 分类挥拍阶段
        phases = classify_swing_phases(frames_data, impact_idx)

        # 4. 生成报告
        report = generate_report(frames_data, phases, impact_idx, impact_vel)

        # 5. 写 CSV
        save_csv(frames_data, phases, smoothed_wy, OUTPUT_CSV)
        print(f"CSV 已保存: {OUTPUT_CSV}")

        # 6. 写 JSON
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"JSON 报告已保存: {OUTPUT_JSON}")

        # 7. 终端摘要
        print("-" * 40)
        print("挥拍分析报告")
        print(f"  蓄力评估:    {report['蓄力评估']}")
        print(f"  击球爆发力:  {report['击球爆发力']}")
        print(f"  动作连贯性:  {report['动作连贯性']}")
        print(f"  最大蓄力角度: {report['_detail']['max_backswing_angle']}°")
        print(f"  跟随帧数:     {report['_detail']['follow_through_frames']}")
        print("=" * 50)

    print("Done.")


if __name__ == "__main__":
    main()
