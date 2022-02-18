import torch.nn.functional as F 
import torch

def eval_result(prediction, ground_truth, desc):
    
    correct_1, correct_10, correct_100 = 0, 0, 0
    
    #_, topk_indices = torch.sort(prediction)
    _, topk_indices = torch.topk(prediction, 1000)
    topk_indices = list(topk_indices)
    top1_indices = topk_indices[:1]
    top10_indices = topk_indices[:10]
    top100_indices = topk_indices[:100]

    if desc:
        rank = [1000]
        for ele in ground_truth:
            if ele in top100_indices:
                correct_100 = 1
            if ele in top10_indices:
                correct_10 = 1
            if ele in top1_indices:
                correct_1 = 1
            if ele in topk_indices:
                rank.append(topk_indices.index(ele))

        pred_rank = min(rank)

    else:
        if ground_truth in top100_indices:
            correct_100 = 1
        if ground_truth in top10_indices:
            correct_10 = 1
        if ground_truth in top1_indices:
            correct_1 = 1
        if ground_truth in topk_indices:
            pred_rank = topk_indices.index(ground_truth)
        else:
            pred_rank = 1000

    return correct_1,correct_10,correct_100, pred_rank