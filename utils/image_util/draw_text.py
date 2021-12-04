import glob

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

landmarks = [' x_0', ' y_0', ' x_1', ' y_1', ' x_2', ' y_2', ' x_3', ' y_3', ' x_4', ' y_4', ' x_5', ' y_5', ' x_6', ' y_6', ' x_7', ' y_7', ' x_8', ' y_8', ' x_9', ' y_9', ' x_10', ' y_10',
             ' x_11', ' y_11', ' x_12', ' y_12', ' x_13', ' y_13', ' x_14', ' y_14', ' x_15', ' y_15', ' x_16', ' y_16', ' x_17', ' y_17', ' x_18', ' y_18', ' x_19', ' y_19', ' x_20', ' y_20',
             ' x_21', ' y_21', ' x_22',
             ' y_22', ' x_23', ' y_23', ' x_24', ' y_24', ' x_25', ' y_25', ' x_26', ' y_26', ' x_27', ' y_27', ' x_28', ' y_28', ' x_29', ' y_29', ' x_30', ' y_30', ' x_31', ' y_31', ' x_32',
             ' y_32', ' x_33', ' y_33',
             ' x_34', ' y_34', ' x_35', ' y_35', ' x_36', ' y_36', ' x_37', ' y_37', ' x_38', ' y_38', ' x_39', ' y_39', ' x_40', ' y_40', ' x_41', ' y_41', ' x_42', ' y_42', ' x_43', ' y_43',
             ' x_44', ' y_44', ' x_45',
             ' y_45', ' x_46', ' y_46', ' x_47', ' y_47', ' x_48', ' y_48', ' x_49', ' y_49', ' x_50', ' y_50', ' x_51', ' y_51', ' x_52', ' y_52', ' x_53', ' y_53', ' x_54', ' y_54', ' x_55',
             ' y_55', ' x_56', ' y_56',
             ' x_57', ' y_57', ' x_58', ' y_58', ' x_59', ' y_59', ' x_60', ' y_60', ' x_61', ' y_61', ' x_62', ' y_62', ' x_63', ' y_63', ' x_64', ' y_64', ' x_65', ' y_65', ' x_66', ' y_66',
             ' x_67', ' y_67']

action_units = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c',
                ' AU45_c']

poses = [' pose_Rx, pose_Ry, pose_Rz,']

exp_dict = {
    'ours': {
        'csv_dir': 'output/paper_model/Ours/processed/0_aligned.csv',
        'img_dir': 'output/paper_model/Ours/may',
        'save_path': 'results/motivation/ours.jpg'
    },
    'makeittalk': {
        'csv_dir': 'output/SOTA/MakeItTalk/processed/MKT_may_aligned.csv',
        'img_dir': 'output/SOTA/MakeItTalk/may',
        'save_path': 'results/motivation/makeittalk.jpg'
    },
    'fomm': {
        'csv_dir': 'output/SOTA/FOMM/FOMM_May_Aligned.csv',
        'img_dir': 'output/SOTA/FOMM/may',
        'save_path': 'results/motivation/fomm.jpg'
    },
    'pc_avs': {
        'csv_dir': 'output/SOTA/PC-AVS/processed/0_aligned.csv',
        'img_dir': 'output/SOTA/PC-AVS/may',
        'save_path': 'results/motivation/pc_avs.jpg'
    },
    'gt': {
        'csv_dir': 'output/SOTA/GT/obama_22s_gt/processed/0_aligned.csv',
        'img_dir': 'output/SOTA/GT/obama_22s_gt/frame',
        'save_path': 'results/motivation/gt.jpg'
    }
}

frame = [138, 232, 51, 188, 200, 303, 545]

for key in exp_dict:
    exp_name = key

    pred_dir = exp_dict[exp_name]['csv_dir']
    gt_dir = 'output/SOTA/GT/obama_22s_gt/processed/0_aligned.csv'
    df_gt, df_pred = pd.read_csv(pred_dir), pd.read_csv(gt_dir)
    pred_au, gt_au = df_pred[action_units].values.tolist(), df_gt[action_units].values.tolist()
    pred_au_np, gt_au_np = np.asarray(pred_au), np.asarray(gt_au)
    pred_landmark, gt_landmark = df_pred[landmarks].values.tolist(), df_gt[landmarks].values.tolist()
    pred_ldk_np, gt_ldk_np = np.asarray(pred_landmark), np.asarray(gt_landmark)

    img_dir = exp_dict[exp_name]['img_dir']
    gt_dir = 'output/SOTA/GT/obama_22s_gt/frame'
    save_path = exp_dict[exp_name]['save_path']

    pred_imgs = glob.glob(f'{img_dir}/*.jpg')
    pred_imgs.sort()
    # print(pred_imgs)
    gt_imgs = glob.glob(f'{gt_dir}/*.jpg')
    gt_imgs.sort()

    result = []
    for i in frame:
        img, gt = pred_imgs[i], gt_imgs[i]
        # [20:360, 80:420]
        image, gt_image = cv2.imread(img), cv2.imread(gt)

        au_error = np.mean(abs(gt_au_np[i] - pred_au_np[i]))
        landmark_error = np.mean(abs(gt_ldk_np[i] - pred_ldk_np[i]))
        # if image.shape[0] != gt_image.shape[0]:
        #     image = cv2.resize(image, (gt_image.shape[0], gt_image.shape[1]))

        image = cv2.resize(image, (256, 256))

        if exp_name != 'gt':
            img_text = cv2.putText(image, text=f'AU error: {round(au_error, 3)}', org=(10, 30), color=(155, 255, 255), thickness=3, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.8)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        result.append(img_rgb)

        if len(result) >= 9:
            break

    result = np.concatenate(result, axis=1)
    # plt.axis('off')
    plt.imshow(result)
    plt.show()

    imageio.imwrite(save_path, result)
