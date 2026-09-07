import cv2
import os

video_path = r"D:\个人资料\实训\belt.mp4"

# 🎯 改用当前项目下的相对路径，避开 Windows 权限拦截
output_folder = "extracted_frames"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"【错误】无法打开视频：{video_path}")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频打开成功！帧率: {fps}")

    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 5 帧保存一张
        if frame_idx % 5 == 0:
            image_name = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")

            # 🔍 核心修改：检查 cv2.imwrite 是否真的写入成功
            success = cv2.imwrite(image_name, frame)
            if success:
                saved_count += 1
            else:
                print(f"警告：第 {frame_idx} 帧保存失败！")

        frame_idx += 1

        # 限制只抽前 300 帧（约10秒）
        if frame_idx >= 300:
            break

    cap.release()
    print(f"🎉 真正的抽帧完成！成功保存了 {saved_count} 张图片。")
