import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, preds, targets, norm_diff=None, eps = 1e-8, epoch=0):

        # assert preds.shape == targets.shape, "Predictions and targets must have same dimensions"

        if not self.window_pred and not isinstance(targets, list):
            preds, targets = preds.T, targets.T # (n_stim, n_samples)
        elif not self.window_pred and isinstance(targets, list):
            preds, targets[0], targets[1] = preds.T, targets[0].T, targets[1].T # (n_stim, n_samples)
        
        # Adjust alpha if necessary
        if self.adjust_alpha:
            alpha = (self.alpha_end - self.alpha_start) * epoch / self.total_epoch
        else:
            alpha = self.alpha_end

        if self.mode == 'mean':
            corr = self._compute_correlation(preds, targets, eps)
            loss = torch.mean(-corr)
            return [loss]
        
        elif self.mode == 'ressamble':
            corr = self._compute_correlation(preds, targets, eps)
            assert preds.shape[0] == 2, "When computing loss on ressamble mode 2 channels must be introduced"
            diff = torch.abs(corr[0] - corr[1])
            corr_loss = torch.mean(-corr)
            loss = corr_loss + alpha * diff
            return [loss, corr, diff]
        
        elif self.mode == 'ild_mae':
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            ild_mae = torch.abs(self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets[0], targets[1]))
            return [ild_mae]
        
        elif self.mode == 'ild_mse':
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            ild_mse = (self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets[0], targets[1]))**2
            return [ild_mse]
        
        elif self.mode == 'diff_mse':
            assert preds.shape[0] == 2, "When computing loss on differnce mode 2 channels must be introduced"
            diff_pred = preds[0] - preds[1]
            diff_target = targets[0] - targets[1]
            diff_mse = torch.mean((diff_pred - diff_target)**2)
            return [diff_mse]
        
        elif self.mode == 'pred_diff_mse':
            assert preds.shape[0] == 1, "When computing loss when predicting diff only 1 channel"
            diff_pred = preds[0]
            diff_target = targets[0] - targets[1]
            diff_mse = torch.mean((diff_pred - diff_target)**2) #mse
            return [diff_mse]
        
        elif self.mode == 'pred_ild_mse': # extract the mean_value
            assert preds.shape[0] == 1, "When computing loss when predicting diff only 1 channel"
            ild_mse = (torch.mean(preds[0]) - self._compute_ild(targets[0], targets[1]))**2
            return [ild_mse]
        
        elif self.mode == 'ild_mse':
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            ild_mse = (self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets[0], targets[1]))**2
            return [ild_mse]
        
        elif self.mode == 'corr_ild_mae':
            corr = self._compute_correlation(preds, targets, eps)
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            ild_mae = torch.abs(self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets[0], targets[1]))
            corr_loss = torch.mean(-corr)
            loss = (1 - alpha) * corr_loss + alpha * ild_mae
            return [loss, corr_loss, ild_mae]
        
        elif self.mode == 'corr_ild_mse':
            corr = self._compute_correlation(preds, targets, eps)
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            ild_mse = (self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets[0], targets[1]))**2
            corr_loss = torch.mean(-corr)
            loss = (1 - alpha) * corr_loss + alpha * ild_mse
            return [loss, corr_loss, ild_mse]
        
        elif self.mode == 'corr_ild_mse_penalty':
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            assert isinstance(targets, list), "When computing penalty loss tagets must include the attended and unattended"
            targets_att = targets[0]
            targets_unatt = targets[1]
            corr = self._compute_correlation(preds, targets_att, eps)
            ild_mse_att = (self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets_att[0], targets_att[1]))**2
            ild_mse_unatt = (self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets_unatt[0], targets_unatt[1]))**2
            ild_mse = 0.5 * (ild_mse_att - ild_mse_unatt)
            corr_loss = torch.mean(-corr)
            loss = (1 - alpha) * corr_loss + alpha * ild_mse
            return [loss, corr_loss, ild_mse]
        
        elif self.mode == 'corr_ild_mse_penalty_w':
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            assert isinstance(targets, list), "When computing penalty loss tagets must include the attended and unattended"            
            targets_att = targets[0]
            targets_unatt = targets[1]
            corr = self._compute_correlation(preds, targets_att, eps)
            ild_mse_att = (self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets_att[0], targets_att[1]))**2
            ild_mse_unatt = (self._compute_ild(preds[0], preds[1]) - self._compute_ild(targets_unatt[0], targets_unatt[1]))**2
            ild_mse = 0.5*(0.8 * ild_mse_att - 0.2 * ild_mse_unatt)
            corr_loss = torch.mean(-corr)
            loss = (1 - alpha) * corr_loss + alpha * ild_mse
            return [loss, corr_loss, ild_mse]
        
        elif self.mode == 'corr_diff_mse':
            corr = self._compute_correlation(preds, targets, eps)
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            diff_pred = preds[0] - preds[1]
            diff_target = targets[0] - targets[1]
            diff_mse = torch.mean((diff_pred - diff_target)**2)
            corr_loss = torch.mean(-corr)
            loss = (1 - alpha) * corr_loss + alpha * diff_mse
            return [loss, corr_loss, diff_mse]
        
        elif self.mode == 'corr_diff_mae':
            corr = self._compute_correlation(preds, targets, eps=1e-8)
            assert preds.shape[0] == 2, "When computing loss on ild mode 2 channels must be introduced"
            diff_pred = preds[0] - preds[1]
            diff_target = targets[0] - targets[1]
            diff_mae = torch.mean(torch.abs(diff_pred - diff_target))
            corr_loss = torch.mean(-corr)
            loss = (1 - alpha) * corr_loss + alpha * diff_mae
            return [loss, corr_loss, diff_mae]
        
        elif self.mode == 'spatial_locus':
            # Use the BCE with logits to convert scalar CNN outputs into 
            loss = F.binary_cross_entropy_with_logits(preds, targets)
            return [loss]
        
        elif self.mode == 'stim_prob':
            # Use the BCE loss to measure the accuracy between the porbs predicted and the true labels
            loss = F.binary_cross_entropy(preds, targets) # [:, 0] as only one label and pred is needed
            return [loss]
        
        elif self.mode == 'triplet_loss':
            # Preds must contain anchor, positive and negative embeddings (eeg, stim1, stim2)
            emb_eeg, emb_stim1, emb_stim2 = preds
            targets = targets.squeeze()
            # Discern the attended and the unattended stim by looking at the targets (labels that contain which one is the attended)
            emb_stima = torch.cat((emb_stim1[targets == 1], emb_stim2[targets == 0]), dim=0)
            emb_stimb = torch.cat((emb_stim1[targets == 0], emb_stim2[targets == 1]), dim=0)
            loss = F.triplet_margin_loss(emb_eeg, emb_stima, emb_stimb)
            return [loss]
        
        else:
            raise ValueError('Introduce a valid loss')

    def _compute_correlation(self, preds, targets, eps=1e-8):
        """
        Compute Pearson correlation coefficient for all channels in a batch.
        Args:
            preds (torch.Tensor): Predicted values (channels x samples).
            targets (torch.Tensor): Target values (channels x samples).
            eps (float): Small value to prevent division by zero.
        Returns:
            torch.Tensor: Correlation coefficients for each channel.
        """
        # Compute means
        preds_mean = preds.mean(dim=1, keepdim=True)
        targets_mean = targets.mean(dim=1, keepdim=True)
        
        # Compute deviations
        preds_dev = preds - preds_mean
        targets_dev = targets - targets_mean
        
        # Compute correlation for all channels simultaneously
        numerator = (preds_dev * targets_dev).sum(dim=1)
        denominator = torch.sqrt((preds_dev**2).sum(dim=1) * (targets_dev**2).sum(dim=1)) + eps
        return numerator / denominator
    
    def _compute_ild(self, left_channel, right_channel, eps=1e-8):
        """
        Compute the interaural level difference (ILD) in dB.
        Args:
            left_channel (torch.Tensor): Left channel data (samples,).
            right_channel (torch.Tensor): Right channel data (samples,).
            eps (float): Small value to prevent division by zero.
        Returns:
            float: ILD in dB.
        """
        # Calculate RMS for both channels
        rms_left = torch.sqrt((left_channel**2).mean()) + eps
        rms_right = torch.sqrt((right_channel**2).mean()) + eps
        
        # Calculate ILD in dB
        return 10 * torch.log10(rms_left / rms_right)