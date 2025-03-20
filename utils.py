from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import mean_squared_error
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

def plot_metrics(metrics:dict):
    plt.figure(figsize=(10,5))
    for key,item in metrics.items():
        plt.plot(item, label=key)

    plt.title('Metrics results')
    plt.xlabel("Epochs")
    plt.ylabel("Metric Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def count_gflops_with_fvcore(model, input_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Calculate FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    gflops = flops.total() / 1e9  # Convert to GFLOPs
    return gflops



def all_metrics(pscores_file, gscores_file, truth_dir):
    # 读取包含预测分数的CSV文件

    df_predicted = pd.read_csv(pscores_file)
    predicted_scores_dict = dict(zip(df_predicted['filename'], df_predicted['score']))

    # 读取包含标准分数的CSV文件
    df_groudtruth = pd.read_csv(gscores_file)

    # 按照标准分数文件中的视频名称顺序来重新排列预测分数列表
    sorted_pscores = [predicted_scores_dict.get(name) for name in df_groudtruth['filename']]
    sorted_gscores = df_groudtruth['score'].tolist()

    SROCC = spearmanr(sorted_gscores, sorted_pscores)[0]  # logit_poor
    PLCC = pearsonr(sorted_gscores, sorted_pscores)[0]

    xlsx = pd.ExcelFile(truth_dir + '/rank-pair-val.xlsx')
    gt_labels_non_source = []
    pr_labels_non_source = []
    gt_labels_source = []
    pr_labels_source = []
    for sheet_pair in xlsx.sheet_names:
        df_pairs = pd.read_excel(xlsx, sheet_name=sheet_pair)
        for index, row in df_pairs.iterrows():
            video_name1 = row.iloc[0]
            video_name2 = row.iloc[1]
            video_rank = row.iloc[2]
            # 获取对应的预测分数
            video_score1 = predicted_scores_dict.get(video_name1)
            video_score2 = predicted_scores_dict.get(video_name2)
            # print("{}_{}\n".format(video_name1,video_name2))
            pred_rank = 1 if video_score1 >= video_score2 else 2

            if sheet_pair == 'nonsource':
                gt_labels_non_source.append(video_rank)
                pr_labels_non_source.append(pred_rank)
            elif sheet_pair == 'source':
                gt_labels_source.append(video_rank)
                pr_labels_source.append(pred_rank)

    acc_non_source = sum(p == l for p, l in zip(gt_labels_non_source, pr_labels_non_source)) / len(gt_labels_non_source)
    acc_source = sum(p == l for p, l in zip(gt_labels_source, pr_labels_source)) / len(gt_labels_source)

    score = 0.45 * SROCC + 0.45 * PLCC + 0.05 * acc_non_source + 0.05 * acc_source
    return score, SROCC, PLCC, acc_non_source, acc_source


def calculate_metrics_model(true_mos_file, pred_mos_file):
    """
    Calculate SROCC, PLCC, KROCC, and RMSE after aligning scores based on filenames.
    Args:
        true_mos_file (str): Path to the CSV file containing ground truth MOS values.
        pred_mos_file (str): Path to the CSV file containing predicted MOS values.
    Returns:
        srocc, plcc, krocc, rmse
    """
    # Read ground truth and predicted scores
    df_true = pd.read_csv(true_mos_file)
    df_pred = pd.read_csv(pred_mos_file)

    # Create dictionaries for quick lookup
    true_scores_dict = dict(zip(df_true['filename'], df_true['score']))
    pred_scores_dict = dict(zip(df_pred['filename'], df_pred['score']))

    # Align predicted scores with ground truth scores
    aligned_true_scores = []
    aligned_pred_scores = []
    for filename in df_true['filename']:
        if filename in pred_scores_dict:
            aligned_true_scores.append(true_scores_dict[filename])
            aligned_pred_scores.append(pred_scores_dict[filename])

    # Convert to numpy arrays
    true_mos = np.array(aligned_true_scores)
    pred_mos = np.array(aligned_pred_scores)
    true_mos = np.array(true_mos).flatten().astype(np.float32)
    pred_mos = np.array(pred_mos).flatten().astype(np.float32)

    # Calculate metrics
    srocc = spearmanr(true_mos, pred_mos)[0]
    plcc = pearsonr(true_mos, pred_mos)[0]
    krocc = kendalltau(true_mos, pred_mos)[0]
    rmse = np.sqrt(mean_squared_error(true_mos, pred_mos))

    return srocc, plcc, krocc, rmse, true_mos, pred_mos


def calculate_metrics(true_mos, pred_mos):
    """
    Calculate SROCC, PLCC, KROCC, and RMSE.
    Args:
        true_mos (list): Ground truth MOS values.
        pred_mos (list): Predicted MOS values.
    Returns:
        srocc, plcc, krocc, rmse
    """
    true_mos = np.array(true_mos).flatten()
    pred_mos = np.array(pred_mos).flatten()

    srocc, _ = spearmanr(true_mos, pred_mos)
    plcc, _ = pearsonr(true_mos, pred_mos)
    krocc, _ = kendalltau(true_mos, pred_mos)
    rmse = np.sqrt(mean_squared_error(true_mos, pred_mos))

    return srocc, plcc, krocc, rmse


