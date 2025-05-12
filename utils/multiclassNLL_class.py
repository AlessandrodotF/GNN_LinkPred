import torch
from torch.nn.functional import log_softmax

class MulticlassNLL(torch.nn.Module):
    """
    A PyTorch module that computes the Negative Log-Likelihood (NLL) loss 
    for a binary classification scenario using softmax.

    This loss function takes positive and negative scores, applies softmax normalization, 
    and computes the negative log-likelihood for the positive class.
    """

    def __init__(self):
        """
        Initializes the MulticlassNLL loss module.
        """
        super(MulticlassNLL, self).__init__()

    def forward(self, pos_scores, neg_scores):
        """ 
        Computes the negative log-likelihood loss.

        Args:
            pos_scores (torch.Tensor): Scores for the positive class.
            neg_scores (torch.Tensor): Scores for the negative class.

        Returns:
            torch.Tensor: The computed negative log-likelihood loss.
        """
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores.unsqueeze(1)], dim=1)
        log_probs = log_softmax(scores, dim=1)
        return -log_probs[:, 0].mean()  # Minimize the negative log-likelihood
