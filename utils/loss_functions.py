import torch
import torch.nn as nn

class CustomLoss(nn.Module):

    """
    
    Params
    ----------

    mode: str Select the loss operating mode:
        - 'mean': return the mean of the computed loss
        - 'ressamble': force the model to have similar coefficients meaning similar behaviour
        - 'ild': add an ild coefficient to the correlation mean of the channels
        - 'ressamble_ild': add an ild coefficient to the correlation mean plus ressamble feature of the channels
    
    window_pred: bool
        when the window is estimated there's no need for transposing dims

    adjust_alpha: bool


    """

    def __init__(self, mode = 'mean', window_pred = False, adjust_alpha = False, alpha_start = 0, alpha_end = 1, total_epoch=20):
        super(CustomLoss, self).__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epoch = total_epoch
        self.mode = mode
        self.adjust_alpha = adjust_alpha
        self.window_pred = window_pred

    def forward(self, preds, targets, eps=1e-8, epoch=0):

        assert preds.shape == targets.shape, "Predictions and targets must have same dimensions"

        if not self.window_pred:
            preds, targets = preds.T, targets.T # (n_stim, n_samples)

        # Adjust alpha if necessary
        if self.adjust_alpha:
            alpha = (self.alpha_end - self.alpha_start) * epoch / self.total_epoch
        else:
            alpha = self.alpha_end

        if self.mode == 'mean':
            corr = self._compute_correlation(preds, targets, eps=eps)
            return torch.mean(-corr)
        
        elif self.mode == 'ressamble':
            corr = self._compute_correlation(preds, targets, eps=eps)
            assert preds.shape[0] == 2, "When computing loss on ressamble mode 2 channels must be introduced"
            diff = torch.abs(corr[0] - corr[1])
            return torch.mean(-corr) + alpha * diff
        
        if self.mode == 'ild':
            corr = self._compute_correlation(preds, targets, eps=eps)
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            diff_ild = torch.abs(self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets[0], targets[1]))
            return torch.mean(-corr) + alpha * diff_ild

    # Compute the pearson correlation coefficient
    def _compute_correlation(self, preds, targets, eps):

        # Compute the correlation for all channels or batches
        n_stim, n_samples = preds.shape
        corr = torch.zeros((n_stim, ))
        for chan, (p, t) in enumerate(zip(preds, targets)):
            vx = p - torch.mean(p)
            vy = t - torch.mean(t)
            corr[chan] = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
        return corr
    
    # Compute the interaural level difference ILD (dB)
    def _compute_ild(self, left_channel, right_channel):
        # Calculate RMS for each channel
        rms_left = torch.sqrt(torch.mean(left_channel**2))
        rms_right = torch.sqrt(torch.mean(right_channel**2))
        # Calculate ILD in dB
        ild = 10 * torch.log10(rms_left / rms_right)
        return ild