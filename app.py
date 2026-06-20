import streamlit as st
import subprocess
import os
import sys
import json
import pandas as pd
import time

st.set_page_config(page_title="🏸 AI 羽毛球挥拍分析仪", layout="wide")

st.title("🏸 AI 羽毛球挥拍分析仪")
st.markdown("上传你的挥拍视频，AI 将自动分析动作质量并给出评估报告。")

# ---- 侧边栏：参数配置 ----
with st.sidebar:
    st.header("⚙️ 参数配置")
    is_left = st.radio("持拍手", ["右手", "左手"], index=0)

    st.divider()
    st.caption("分析指标：肘关节角度、手腕速度、动作阶段")

# ---- 主区域：上传视频 ----
uploaded_file = st.file_uploader(
    "📤 上传挥拍视频",
    type=["mp4", "mov", "avi", "mkv"],
    help="支持 MP4 / MOV / AVI / MKV 格式"
)

if uploaded_file:
    # 保存上传的视频
    input_path = "input.mp4"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(input_path)

    # 视频信息
    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    st.caption(f"文件大小: {file_size_mb:.1f} MB")

    if st.button("🚀 开始 AI 分析", type="primary", use_container_width=True):
        output_video = "output_processed.mp4"
        output_csv = "swing_data.csv"
        output_json = "swing_report.json"

        # 删除上次运行的旧文件
        for f in [output_video, output_csv, output_json]:
            if os.path.exists(f):
                os.remove(f)

        progress_bar = st.progress(0, text="正在初始化模型...")
        status_area = st.empty()

        # 构造命令：通过 --headless 运行分析
        cmd = [
            sys.executable, "pose_test.py",
            "--input", input_path,
            "--output", output_video,
            "--csv", output_csv,
            "--json", output_json,
            "--headless",
        ]
        # 如果用户选了左手，需要修改 IS_LEFT_HANDED
        # 通过临时设置环境变量的方式传递给脚本
        env = os.environ.copy()
        if is_left == "左手":
            env["BADMINTON_LEFT_HANDED"] = "1"

        try:
            progress_bar.progress(10, text="正在逐帧分析姿态...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            # 显示终端日志
            with st.expander("📋 处理日志", expanded=False):
                st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
                if result.stderr:
                    st.code(result.stderr, language="bash")

            if result.returncode != 0:
                st.error(f"分析程序异常退出 (code {result.returncode})")
            else:
                progress_bar.progress(80, text="正在生成报告...")

                # ---- 展示结果 ----
                tab1, tab2, tab3 = st.tabs(["📊 评估报告", "🎬 标注视频", "📈 逐帧数据"])

                with tab1:
                    st.subheader("挥拍质量评估")
                    if os.path.exists(output_json):
                        with open(output_json, "r", encoding="utf-8") as f:
                            report = json.load(f)

                        # 三项核心指标用大卡片展示
                        col1, col2, col3 = st.columns(3)
                        verdict_color = {
                            "优秀": "#4CAF50",
                            "高": "#4CAF50",
                            "完整": "#4CAF50",
                        }

                        with col1:
                            val = report.get("蓄力评估", "—")
                            bg = verdict_color.get(val, "#FF9800")
                            st.markdown(
                                f"<div style='background:{bg};padding:20px;border-radius:10px;text-align:center'>"
                                f"<h4 style='color:white;margin:0'>蓄力评估</h4>"
                                f"<h2 style='color:white;margin:5px'>{val}</h2>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if "_detail" in report:
                                st.metric(
                                    "最大蓄力角度",
                                    f"{report['_detail']['max_backswing_angle']}°",
                                )

                        with col2:
                            val = report.get("击球爆发力", "—")
                            bg = verdict_color.get(val, "#FF9800")
                            st.markdown(
                                f"<div style='background:{bg};padding:20px;border-radius:10px;text-align:center'>"
                                f"<h4 style='color:white;margin:0'>击球爆发力</h4>"
                                f"<h2 style='color:white;margin:5px'>{val}</h2>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if "_detail" in report:
                                st.metric(
                                    "手腕瞬时速度",
                                    f"{report['_detail']['impact_velocity_px_per_frame']} px/frame",
                                )

                        with col3:
                            val = report.get("动作连贯性", "—")
                            bg = verdict_color.get(val, "#FF9800")
                            st.markdown(
                                f"<div style='background:{bg};padding:20px;border-radius:10px;text-align:center'>"
                                f"<h4 style='color:white;margin:0'>动作连贯性</h4>"
                                f"<h2 style='color:white;margin:5px'>{val}</h2>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if "_detail" in report:
                                st.metric(
                                    "随挥帧数",
                                    report["_detail"]["follow_through_frames"],
                                )

                        st.divider()
                        st.json(report)
                    else:
                        st.warning("未找到报告文件")

                with tab2:
                    st.subheader("AI 标注视频")
                    if os.path.exists(output_video):
                        st.video(output_video)
                        # 提供下载
                        with open(output_video, "rb") as f:
                            st.download_button(
                                "⬇️ 下载标注视频",
                                f,
                                file_name="badminton_analyzed.mp4",
                                mime="video/mp4",
                            )
                    else:
                        st.warning("标注视频未生成，请检查日志。")

                with tab3:
                    st.subheader("逐帧分析数据")
                    if os.path.exists(output_csv):
                        df = pd.read_csv(output_csv)
                        st.dataframe(df, use_container_width=True, height=400)

                        # 下载
                        st.download_button(
                            "⬇️ 下载 CSV",
                            df.to_csv(index=False).encode("utf-8"),
                            file_name="swing_data.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("CSV 数据文件未生成。")

                progress_bar.progress(100, text="✅ 分析完成！")

        except subprocess.TimeoutExpired:
            st.error("⏱️ 分析超时（超过 10 分钟），请尝试更短的视频。")
        except Exception as e:
            st.error(f"运行出错: {e}")

else:
    # 未上传时的欢迎页面
    st.info("👈 请先上传一段羽毛球挥拍视频（建议 3-10 秒，画面清晰，单人挥拍）。")

    with st.expander("📖 使用说明", expanded=True):
        st.markdown("""
        ### 如何使用
        1. **录制视频**：用手机拍摄自己的挥拍动作（侧面视角最佳）
        2. **上传视频**：点击上方上传按钮，选择视频文件
        3. **开始分析**：点击「开始 AI 分析」按钮
        4. **查看报告**：AI 会给出蓄力、爆发力、连贯性三项评分

        ### 分析指标说明
        | 指标 | 含义 | 优秀标准 |
        |------|------|----------|
        | 蓄力评估 | 挥拍准备阶段肘关节展开幅度 | 最大角度 > 150° |
        | 击球爆发力 | 击球瞬间手腕下压速度 | > 10 px/frame |
        | 动作连贯性 | 击球后随挥动作完整性 | 随挥帧数 > 20 |

        ### ⚠️ 注意事项
        - 视频中应只有 **一个人** 在挥拍
        - 确保人物全身可见，光线充足
        - 侧面拍摄效果最佳（相机与挥拍方向平行）
        """)
