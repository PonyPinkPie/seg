import numpy as np
import cv2
from prettytable import PrettyTable

epsilon = 1e-6


def get_contour_points_from_mask(mask, sorted_by_area=False):
    """
    根据mask获取所有的边缘点
    """
    if cv2.__version__[0] == "4":
        contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_contours = len(contours)

    if sorted_by_area:
        contours = sorted(
            contours, key=lambda c: cv2.contourArea(c), reverse=True)

    all_polygon_points = []
    for idx_c in range(num_contours):
        contour = contours[idx_c]
        if hierarchy is None or hierarchy[0][idx_c][3] != -1:
            continue

        pts = [[int(pt[0][0]), int(pt[0][1])] for pt in contour]
        if len(pts) == 0:
            continue
        all_polygon_points.append(pts)

    return all_polygon_points


def calculate_polygon_iou(polygon1, polygon2):
    pts1_x, pts1_y = [pt[0] for pt in polygon1], [pt[1] for pt in polygon1]
    pts2_x, pts2_y = [pt[0] for pt in polygon2], [pt[1] for pt in polygon2]
    pts1_x_min, pts1_y_min = min(pts1_x), min(pts1_y)
    pts1_x_max, pts1_y_max = max(pts1_x), max(pts1_y)
    pts2_x_min, pts2_y_min = min(pts2_x), min(pts2_y)
    pts2_x_max, pts2_y_max = max(pts2_x), max(pts2_y)

    if pts1_x_min > pts2_x_max or pts2_x_min > pts1_x_max or pts1_y_min > pts2_y_max or pts2_y_min > pts1_y_max:
        return 0.0

    x_min = max(pts1_x_min, pts2_x_min)
    y_min = max(pts1_y_min, pts2_y_min)
    x_max = max(pts1_x_max, pts2_x_max)
    y_max = max(pts1_y_max, pts1_y_max)

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    image1 = np.zeros((height, width), np.uint8)
    image2 = np.zeros((height, width), np.uint8)

    polygon1 = np.asarray(polygon1)
    polygon2 = np.asarray(polygon2)

    polygon1 -= [[x_min, y_min]]
    polygon2 -= [[x_min, y_min]]

    mask1 = cv2.fillPoly(image1, [polygon1], 1)
    mask2 = cv2.fillPoly(image2, [polygon2], 1)

    mask_and = np.bitwise_and(mask1, mask2)
    area_and = np.sum(mask_and).astype(np.float32)
    if area_and == 0:
        return 0.0

    mask_or = np.bitwise_or(mask1, mask2)
    area_or = np.sum(mask_or).astype(np.float32)
    iou = area_and / area_or
    return iou


def confuse_matrix_for_segmentation(
        predict_prob, gt_label,
        min_iou_threshold=0.35,
        min_score_threshold=0.05,
        max_score_threshold=1.0,
        score_step=0.05,
        with_morphology=True
):
    threshold_list = np.arange(min_score_threshold, max_score_threshold, score_step)
    gt_polygon_points = get_contour_points_from_mask(gt_label)
    num_thres = threshold_list.shape[0]
    tp_list = np.zeros(num_thres, dtype=np.int32)
    fp_list = np.zeros(num_thres, dtype=np.int32)
    fn_list = np.zeros(num_thres, dtype=np.int32)
    for thres_idx, threshold in enumerate(list(threshold_list)):
        pred_label = np.array(predict_prob >= threshold, dtype=np.uint8) * 255

        if with_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            pred_label = cv2.morphologyEx(pred_label, cv2.MORPH_CLOSE, kernel)

        pred_polygon_points = get_contour_points_from_mask(pred_label)
        match_gt = [False] * len(gt_polygon_points)
        match_pred = [False] * len(pred_polygon_points)

        for gt_idx, gt_polygon_point in enumerate(gt_polygon_points):
            for pred_idx, pred_polygon_point in enumerate(pred_polygon_points):
                iou = calculate_polygon_iou(gt_polygon_point, pred_polygon_point)
                if iou < min_iou_threshold:
                    continue
                match_gt[gt_idx] = True
                match_pred[pred_idx] = True
            tp_list[thres_idx] = sum(match_gt)
            fp_list[thres_idx] = max(0, len(match_pred) - sum(match_gt))
            fn_list[thres_idx] = len(match_gt) - sum(match_gt)
    return tp_list, fp_list, fn_list, threshold_list


def get_pr_point(tp_list, fp_list, fn_list):
    precision_list = tp_list / (tp_list + fp_list + epsilon)
    recall_list = tp_list / (tp_list + fn_list + epsilon)
    return precision_list, recall_list


def get_seg_det_roc_point(tp_list, fp_list, fn_list, num_gt):
    tpr_list = tp_list / (tp_list + fn_list + epsilon)
    return tpr_list, fp_list / max(num_gt, 1)


def get_f1_point(precision_list, recall_list):
    return [2 * p * r / (p + r + epsilon) for p, r in zip(precision_list, recall_list)]


def area_under_curve(x, y):
    area = 0
    last_idx = 0
    for i in range(1, len(x)):
        if x[i] > x[last_idx]:
            continue
        area += (y[i] + y[last_idx]) * (x[last_idx] - x[i]) / 2
        last_idx = i
    return area


def calculate_iou(y_pred, y_true):
    intersection = np.sum(np.bitwise_and(y_pred, y_true))
    union = np.sum(np.bitwise_or(y_pred, y_true))
    iou = intersection / (union + epsilon)
    return np.round(iou, 4)


def calculate_auc(tpr_list, fpr_list):
    tpr = np.append(np.insert(tpr_list, 0, 1.0), 0.0)
    fpr = np.append(np.insert(fpr_list, 0, 1.0), 0.0)
    auc = area_under_curve(fpr, tpr)
    return np.round(auc, 4)


def calculate_aupr(precision_list, recall_list):
    prec = np.append(np.insert(precision_list, 0, 0.0), 1.0)
    rec = np.append(np.insert(recall_list, 0, 1.0), 0.0)
    aupr = area_under_curve(rec, prec)

    return np.round(aupr, 4)


def calculate_metric_for_more(prob, pred, mask):
    tp_list, fp_list, fn_list, threshold_list = confuse_matrix_for_segmentation(prob, mask)
    precision_list, recall_list = get_pr_point(tp_list, fp_list, fn_list)
    tpr, fpr = get_seg_det_roc_point(tp_list, fp_list, fn_list, len(get_contour_points_from_mask(mask)))
    f1_list = get_f1_point(precision_list, recall_list)
    auc = calculate_auc(tpr, fpr)
    aupr = calculate_aupr(precision_list, recall_list)
    iou = calculate_iou(pred, mask)
    prf1 = np.stack([precision_list, recall_list, f1_list], axis=0)
    iou_auc_aupr = [iou, auc, aupr]
    tp_fp_fn = np.stack([tp_list, fp_list, fn_list], axis=0)
    return [tp_fp_fn, prf1, iou_auc_aupr], threshold_list


def calculate_intersection_and_union(y_pred, y_true):
    intersection = np.sum(np.bitwise_and(y_pred, y_true))
    union = np.sum(np.bitwise_or(y_pred, y_true)) - intersection
    return intersection, union


def calculate_metric_for_one(prob, mask):
    tp_list, fp_list, fn_list, threshold_list = confuse_matrix_for_segmentation(prob, mask)
    precision_list, recall_list = get_pr_point(tp_list, fp_list, fn_list)
    tpr, fpr = get_seg_det_roc_point(tp_list, fp_list, fn_list, len(get_contour_points_from_mask(mask)))
    f1_list = get_f1_point(precision_list, recall_list)
    auc = calculate_auc(tpr, fpr)
    aupr = calculate_aupr(precision_list, recall_list)
    iou = [calculate_intersection_and_union(prob > t, mask) for t in threshold_list]
    iou = list(zip(*iou))
    iou_auc_aupr = [iou, auc, aupr]
    prf1 = np.stack([precision_list, recall_list, f1_list], axis=0)
    tp_fp_fn = np.stack([tp_list, fp_list, fn_list], axis=0)
    return [tp_fp_fn, prf1, iou_auc_aupr], threshold_list


def parse_seg_metrics(all_metrics):
    all_tp_fp_fn_dict = dict()
    all_prf1_dict = dict()
    all_iou_auc_aupro_dict = dict()
    for class_name in all_metrics:
        all_tp_fp_fn_dict[class_name] = [m[0] for m in all_metrics[class_name]]
        all_prf1_dict[class_name] = [m[1] for m in all_metrics[class_name]]
        all_iou_auc_aupro_dict[class_name] = [m[2] for m in all_metrics[class_name]]

    all_tp_fp_fn = list(all_tp_fp_fn_dict.values())
    all_prf1 = list(all_prf1_dict.values())
    all_iou_auc_aupro = list(all_iou_auc_aupro_dict.values())

    categories_sum_tp_fp_fn = [np.array(a).sum(axis=0) for a in all_tp_fp_fn]
    categories_avg_prf1 = [np.array(a).mean(axis=0) for a in all_prf1]
    categories_avg_iou_aoc_aupr = [np.array(a).mean(axis=0) for a in all_iou_auc_aupro]

    avg_all_prf1 = np.array(categories_avg_prf1).mean(axis=0, keepdims=True)
    best_index = avg_all_prf1[..., -1, :].argmax()  # F1最佳阈值
    curr_avg_all_prf1 = avg_all_prf1[..., best_index]

    curr_best_categories_tp_fp_fn = [a[..., best_index] for a in categories_sum_tp_fp_fn]
    curr_best_categories_prf1 = [a[..., best_index] for a in categories_avg_prf1]

    curr_best_categories_metric = np.concatenate(
        [curr_best_categories_tp_fp_fn, curr_best_categories_prf1, categories_avg_iou_aoc_aupr], axis=1)

    curr_sum_all_tp_fp_fn = np.array(curr_best_categories_tp_fp_fn).sum(axis=0, keepdims=True)
    curr_avg_all_iou_aoc_aupr = np.array(categories_avg_iou_aoc_aupr).mean(axis=0, keepdims=True)

    curr_avg_all_metric = np.concatenate([curr_sum_all_tp_fp_fn, curr_avg_all_prf1, curr_avg_all_iou_aoc_aupr], axis=1)

    curr_metrics = np.concatenate([curr_best_categories_metric, curr_avg_all_metric], axis=0).round(4)
    return curr_metrics, best_index


def parse_seg_metrics_to_table(curr_metrics, best_metrics, label2class, logger):
    table = PrettyTable()
    table.field_names = ["", "TP", "FP", "FN", "Precision", "Recall", "F1", "iou", "AUC", "AUPR"]
    for i, metrics in enumerate(curr_metrics[:-1]):
        table.add_row([label2class[i]] + metrics[:3].astype(np.int32).tolist() + metrics[3:].tolist())
    table.add_row(["curr_metrics"] + curr_metrics[-1][:3].astype(np.int32).tolist() + curr_metrics[-1][3:].tolist())
    table.add_row(["best_metrics"] + best_metrics[:3].astype(np.int32).tolist() + best_metrics[3:].tolist())

    msgs = table.__str__().split('\n')
    for msg in msgs:
        logger.info(msg)
