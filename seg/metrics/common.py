import numpy as np
import cv2


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

    mask1 = cv2.fillPoly(image1, [polygon1], 1)
    mask2 = cv2.fillPoly(image2, [polygon2], 1)

    mask_and = np.bitwise_and(mask1, mask2)
    area_and = np.sum(mask_and)
    if area_and == 0:
        return 0.0

    mask_or = np.bitwise_or(mask1, mask2)
    area_or = np.sum(mask_or)
    iou = float(area_and) / float(area_or)
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
    tp_list = np.zeros((num_thres), dtype=np.int32)
    fp_list = np.zeros((num_thres), dtype=np.int32)
    fn_list = np.zeros((num_thres), dtype=np.int32)
    tn_list = np.zeros((num_thres), dtype=np.int32)
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
            fp_list[thres_idx] = len(match_pred) - sum(match_gt)
            fn_list[thres_idx] = len(match_gt) - sum(match_gt)
            tn_list = 0
    return tp_list, fp_list, fn_list, tn_list, threshold_list


def get_pr_pont(tp_list, fp_list, fn_list):
    precision_list = tp_list / (tp_list + fp_list + 1e-6)
    recall_list = tp_list / (tp_list + fn_list + 1e-6)
    return precision_list, recall_list


def get_seg_det_roc_point(tp_list, fp_list, fn_list, num_gt):
    tpr_list = tp_list / (tp_list + fn_list + 1e-6)
    return tpr_list, fp_list / max(num_gt, 1)


def get_f1_point(precision_list, recall_list):
    return [2 * p * r / (p + r + 1e-6) for p, r in zip(precision_list, recall_list)]


def area_under_curve(x, y):
    area = 0
    last_idx = 0
    for i in range(1, len(x)):
        if x[i] > x[last_idx]:
            continue
        area += (y[i] + y[last_idx]) * (x[last_idx] - x[i]) / 2
        last_idx = i
    return area


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



# def get_cls_roc_point(tp_list, fp_list, fn_list, tn_list):
#     tpr_list = tp_list / (tp_list + fn_list + 1e-6)
#     fpr_list = tp_list / (fp_list + tn_list + 1e-6)
#     return tpr_list, fpr_list
#
