import os
import argparse
import torch
import numpy as np
from skimage import io, morphology, color

def computeF1(pred, gt):
    tp = (gt * pred).sum().float()
    fp = ((1 - gt) * pred).sum().float()
    fn = (gt * (1 - pred)).sum().float()
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)

    return f1_score * 100, precision * 100, recall * 100

def computeTopo(pred, gt):
    pred = morphology.skeletonize(pred >= 0.5)
    gt = morphology.skeletonize(gt >= 0.5)

    cor_tp = np.sum(gt & pred)
    com_tp = np.sum(gt & pred)

    sk_pred_sum = np.sum(pred)
    sk_gt_sum = np.sum(gt)

    smooth = 1e-7
    correctness = cor_tp / (sk_pred_sum + smooth)
    completeness = com_tp / (sk_gt_sum + smooth)
    quality = cor_tp / (sk_pred_sum + sk_gt_sum - com_tp + smooth)

    return correctness * 100, completeness * 100, quality * 100

class Evaluator:
    def __init__(self, pred_dir, gt_dir, threshold=0.5):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.threshold = threshold

    def load_mask(self, path):
        mask = io.imread(path)
        if mask.ndim > 2:
            mask = color.rgb2gray(mask)
        return torch.tensor(mask, dtype=torch.float32)

    def compute_f1_and_topo(self, pred_mask, gt_mask):
        pred_mask = (pred_mask > self.threshold).float()
        gt_mask = (gt_mask > self.threshold).float()
        f1, precision, recall = computeF1(pred_mask, gt_mask)
        correctness, completeness, quality = computeTopo(pred_mask[0].detach().cpu().numpy().astype(int),
                                                         gt_mask[0].detach().cpu().numpy().astype(int))
        return f1, precision, recall, correctness, completeness, quality

    def evaluate(self):
        pred_files = sorted(os.listdir(self.pred_dir))
        gt_files = sorted(os.listdir(self.gt_dir))

        num_files = len(pred_files)
        metrics_sum = np.zeros(6)

        for pred_file, gt_file in zip(pred_files, gt_files):
            pred_path = os.path.join(self.pred_dir, pred_file)
            gt_path = os.path.join(self.gt_dir, gt_file)

            pred_mask = self.load_mask(pred_path)
            gt_mask = self.load_mask(gt_path)

            if pred_mask.shape != gt_mask.shape:
                raise AssertionError(f"Shape mismatch: {pred_path}, {gt_path}")

            pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)

            metrics = self.compute_f1_and_topo(pred_mask, gt_mask)
            metrics_sum += np.array(metrics)

            print(f"File: {pred_file}")
            print(f"F1 Score: {metrics[0]:.2f}, Precision: {metrics[1]:.2f}, Recall: {metrics[2]:.2f}")
            print(f"Correctness: {metrics[3]:.2f}, Completeness: {metrics[4]:.2f}, Quality: {metrics[5]:.2f}")
            print("-" * 50)

        avg_metrics = metrics_sum / num_files
        print("Overall Metrics:")
        print(f"Average F1 Score: {avg_metrics[0]:.2f}")
        print(f"Average Precision: {avg_metrics[1]:.2f}")
        print(f"Average Recall: {avg_metrics[2]:.2f}")
        print(f"Average Correctness: {avg_metrics[3]:.2f}")
        print(f"Average Completeness: {avg_metrics[4]:.2f}")
        print(f"Average Quality: {avg_metrics[5]:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks.")
    parser.add_argument('--pred_dir', type=str, default='./results', help='Directory containing predicted masks')
    parser.add_argument('--gt_dir', type=str, default='./mask', help='Directory containing ground-truth masks')
    args = parser.parse_args()

    evaluator = Evaluator(pred_dir=args.pred_dir, gt_dir=args.gt_dir)
    evaluator.evaluate()

if __name__ == "__main__":
    main()

