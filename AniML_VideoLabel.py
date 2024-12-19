import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd


def LabelVideo(VideoName, Folder,
               OutputFolder, BehaviorLabels,
               InputFileType='.avi',
               OutputFileType='.avi',
               FrameCount=True,
               Resolution_factor=1,
               fromFrame=0,
               toFrame=np.inf,
               pix_threshold=[],
               only_pix_threshold=False,
               colormap='Reds',
               LabelType='Text',
               plot=False,
               inIncrement=1):  # New parameter for frame increment

    # Create OutputFolder if it doesn't exist
    if not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder)
        print(f"Output folder '{OutputFolder}' created.")

    print("Reading " + Folder + '/' + VideoName + InputFileType)
    video = cv2.VideoCapture(Folder + '/' + VideoName + InputFileType)
    video.set(cv2.CAP_PROP_POS_FRAMES, fromFrame)

    if np.isinf(toFrame):
        toFrame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'from frame {fromFrame} to frame {toFrame} out of {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} frames')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_count = fromFrame
    count = 0

    new_width = round(width * Resolution_factor)
    new_height = round(height * Resolution_factor)
    summary_width = int(new_width/2)
    plot_width = new_width
    plot_area_height = new_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Calculate the total width for the output video
    total_width = new_width + (plot_width if plot else 0) + summary_width
    out = cv2.VideoWriter(
        OutputFolder + '/' + VideoName + '_Labeled_' + datetime.now().strftime("%H%M") + OutputFileType,
        fourcc, fps,
        (total_width, new_height))
    print("Writing to " + OutputFolder + '/' + VideoName
          + '_Labeled_' + datetime.now().strftime("%H%M") + OutputFileType)

    behaviors = BehaviorLabels.columns.tolist()
    behavior_durations = {behavior: 0 for behavior in behaviors}

    while True and frame_count <= toFrame:
        print("\r" + f"Frame: {fromFrame + count}/{len(range(fromFrame, toFrame, inIncrement))}", end='', flush=True)
        ret, frame = video.read()
        if frame_count % inIncrement != 0:
            frame_count += inIncrement
            continue

        if (frame is None and count == 0) or (frame is None and count == toFrame):
            break
        count += 1

        if pix_threshold:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            above_threshold_mask = gray_frame > pix_threshold
            colormap = plt.get_cmap(colormap)
            normalized_values = gray_frame[above_threshold_mask] / 255.0
            colormap_values = colormap(normalized_values)[:, :3]
            colormap_values_bgr = (colormap_values[:, ::-1] * 255).astype(np.uint8)

            if only_pix_threshold:
                frame[:] = 0

            frame[above_threshold_mask] = colormap_values_bgr

        if not ret:
            break

        if FrameCount:
            cv2.putText(frame, f"{frame_count}", (int(8.5 * width / 10), int(9 * height / 10)), font, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # Add behavior labels
        if frame_count < len(BehaviorLabels):
            for i, behavior in enumerate(behaviors):
                if BehaviorLabels[behavior].iloc[frame_count] == 1:
                    y_pos = int((1 + i * 0.5) * height / 10)
                    cv2.putText(frame, behavior, (int(width / 10), y_pos), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    behavior_durations[behavior] += 1 / fps

        frame = cv2.resize(frame, (new_width, new_height))

        # Add behavior summary on the right
        summary_area = np.zeros((new_height, summary_width, 3), dtype=np.uint8)
        for i, behavior in enumerate(behaviors):
            duration_text = f"{behavior}: {behavior_durations[behavior]:.1f}s"
            cv2.putText(summary_area, duration_text, (10, int((i + 1) * 10)), font, 0.2, (255, 255, 255), 1,
                        cv2.LINE_AA)

        combined_frame = np.hstack((frame, summary_area))

        # Plot behavior labels onto the plot area with traces
        if plot:
            plot_area = np.zeros((plot_area_height, plot_width, 3), dtype=np.uint8)

            x_scale_factor = plot_width / max((toFrame - fromFrame) // inIncrement, 1)

            for i, behavior in enumerate(behaviors):
                normalized_behavior = BehaviorLabels[behavior].values.astype(float) / BehaviorLabels[behavior].max()
                for j in range(1, count):
                    if j < len(normalized_behavior):
                        x_prev = int((j - 1) * x_scale_factor)
                        x_curr = int(j * x_scale_factor)
                        y_prev = int(plot_area_height - normalized_behavior[j - 1] * (plot_area_height - 10))
                        y_curr = int(plot_area_height - normalized_behavior[j] * (plot_area_height - 10))
                        color = (0, 0, 255) if i % 2 == 0 else (0, 255, 0)
                        cv2.line(plot_area, (x_prev, y_prev), (x_curr, y_curr), color, 1)

            combined_frame = np.hstack((combined_frame, plot_area))

        out.write(combined_frame)

        frame_count += 1

    video.release()
    out.release()
    cv2.destroyAllWindows()
    print('\nDone writing video.')
