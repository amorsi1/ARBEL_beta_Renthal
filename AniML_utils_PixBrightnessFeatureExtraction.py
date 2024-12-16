"""Utility functions for extracting brightness features from a behavior video
take chosen body parts locations x,y cords (bp_xcords, bp_ycord).
returns pixel intensity-related measurements around the chose body parts.
"""
from AniML_utils_GeneralFunctions import *
import time
import gc

def depth_pixels(pose_data_file, video_file_path, bp_list, square_size, pix_threshold=[], scale_x=1, scale_y=1, create_video=False, min_prob = 0.8):
    # if only one value was given as size
    if np.size(square_size)==1:
        size=np.ones(np.size(bp_list))*square_size
    startTime_for_tictoc = time.time()
    label = open_file_as_dataframe(pose_data_file)
    the_video = cv2.VideoCapture(video_file_path)
    frame_ht = int(the_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_wt = int(the_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_of_frames = int(the_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = the_video.get(cv2.CAP_PROP_FPS)
    print(f'video length is {num_of_frames / fps / 60} mins')
    # Create an empty list to store the frames
    frames = []

    BPratios_all = []

    # Pre-compute the pixel threshold from the first frame
    ret, first_frame = the_video.read()
    if not ret:
        return pd.DataFrame()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    min_val, max_val = cv2.minMaxLoc(first_frame)[0:2]
    if pix_threshold == []:
        pix_threshold = (max_val - min_val) / 2
        print(f'Video brightness detection threshold set to mean({min_val},{max_val}): {pix_threshold}')
    else:
        pix_threshold = (max_val - min_val) * pix_threshold if pix_threshold < 1 else pix_threshold
        print(f'Video brightness detection threshold set to {pix_threshold}% of max-min values')

    i_frame=0
    while True:
        ret, frame = the_video.read()
        if not ret:
            break
        #process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if scale_x!=1 or scale_y!=1:
            scaled_width = int(frame.shape[1] * x_scale)
            scaled_height = int(frame.shape[0] * y_scale)
            frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

        frame[frame < pix_threshold] = 1 # Black out below threshold (dont use 0 because 0/0 is nan)

        bp_data = {}
        for i, bp1 in enumerate(bp_list):
            for j, bp2 in enumerate(bp_list):
                if j > i:
                    # Get body part coordinates and probabilities
                    x1, y1, prob1 = int(label[bp1 + '_x'].values[i_frame]), int(label[bp1 + '_y'].values[i_frame]), \
                    label[bp1 + '_prob'].values[i_frame]
                    x2, y2, prob2 = int(label[bp2 + '_x'].values[i_frame]), int(label[bp2 + '_y'].values[i_frame]), \
                    label[bp2 + '_prob'].values[i_frame]

                    # Define square areas, ensuring bounds stay within the frame
                    x1_min, x1_max = int(max(0, x1 - square_size[i]*scale_x // 2)), int(min(frame_wt, x1 + square_size[i]*scale_x // 2))
                    y1_min, y1_max = int(max(0, y1 - square_size[i]*scale_y // 2)), int(min(frame_ht, y1 + square_size[i]*scale_y // 2))
                    x2_min, x2_max = int(max(0, x2 - square_size[j]*scale_x // 2)), int(min(frame_wt, x2 + square_size[j]*scale_x // 2))
                    y2_min, y2_max = int(max(0, y2 - square_size[j]*scale_y // 2)), int(min(frame_ht, y2 + square_size[j]*scale_y // 2))

                    # Compute mean pixel intensity, respecting probability thresholds
                    bp1_mean_pixels = np.mean(frame[y1_min:y1_max, x1_min:x1_max]) if prob1 >= min_prob else 1
                    bp2_mean_pixels = np.mean(frame[y2_min:y2_max, x2_min:x2_max]) if prob2 >= min_prob else 1
                    bp_data[f'Pix_{bp1}'] = bp1_mean_pixels
                    bp_data[f'Pix_{bp2}'] = bp2_mean_pixels
                    bp_data[f'Log10(Pix_{bp1}/Pix_{bp2})'] = np.log10(bp1_mean_pixels / bp2_mean_pixels)

        BPratios_all.append(bp_data)

        if create_video:
            frames.append(frame)

        i_frame = i_frame + 1

    # Convert list of bp_data to DataFrame after the loop
    BPratios_all = pd.DataFrame(BPratios_all)
    print("Elapsed time is " + str(round(time.time() - startTime_for_tictoc, 2)) + " seconds.")


    if create_video==True:
        print('Creating image brightness video...')
        output_file_name = os.path.splitext(video_file_path)[0] + '_PixThreshold(' + str(pix_threshold) +')_' + datetime.now().strftime("%H%M") + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        out = cv2.VideoWriter(output_file_name, fourcc, fps, (frame_wt, frame_ht))
        i_frame=0
        for frame in frames:
            print(i_frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            thickness = 1  # Thickness of the square border
            color = (0, 255, 0)  # Green color in BGR format
            for i, bp1 in enumerate(bp_list):
                for j, bp2 in enumerate(bp_list):
                    x, y = (int(label[bp1 + '_x'].values[i_frame]), int(label[bp1 + '_y'].values[i_frame]))
                    if label[bp1 + '_prob'].values[i_frame] > min_prob:
                        cv2.rectangle(frame, (x-square_size[i]//2, y-square_size[i]//2), (x+square_size[i]//2, y+square_size[i]//2), color, thickness)
                    x, y = (int(label[bp2 + '_x'].values[i_frame]), int(label[bp2 + '_y'].values[i_frame]))
                    if label[bp2 + '_prob'].values[i_frame] > min_prob:
                        cv2.rectangle(frame, (x-square_size[j]//2, y-square_size[j]//2), (x+square_size[j]//2, y+square_size[j]//2), color, thickness)
            out.write(frame)
            i_frame=i_frame+1
        out.release()
        print('Video created: ' + output_file_name )

    # Release the video capture object
    the_video.release()
    del the_video
    gc.collect()  # garbage collect

    return BPratios_all


def PixBrightFeatureExtract(pose_data_file, video_file_path, bp_list, square_size, pix_threshold, dt_vel=2, scale_x=1, scale_y=1, create_video=False, min_prob=0.80):

    bppixels_ratio = depth_pixels(pose_data_file, video_file_path, bp_list, square_size, pix_threshold, scale_x,
                                  scale_y, create_video, min_prob)

    # Absolute first derivative before
    bppixels_ratio_1diffBefore = bppixels_ratio.diff(periods=dt_vel).abs()
    bppixels_ratio_1diffBefore.columns = [f"|d/dt({col})|" for col in bppixels_ratio.columns]
    X = pd.concat([bppixels_ratio,
                   bppixels_ratio_1diffBefore,
                   ], axis=1)
    return X

# Not used
    ## Peak frequency
    # peak_frequency = bppixels_ratio.apply(lambda column: peak_freq(column, 10, 1))
    # peak_frequency.columns = [f'{col}_freq' for col in bppixels_ratio.columns]

    ## Absolute second derivative before
    # bppixels_ratio_2diffBefore = bppixels_ratio.diff(periods=dt_before).diff(periods=1).abs()
    # bppixels_ratio_2diffBefore.columns = [f"|d2/dt2({col})|" for col in bppixels_ratio.columns]