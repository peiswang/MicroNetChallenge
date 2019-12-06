from torch import nn
from torch.nn import functional as F


class ModelRefineryWrapper(nn.Module):
    """Convenient wrapper class to train a model with a label refinery."""

    def __init__(self, label_refinery):
        super().__init__()
        self.label_refinery = label_refinery

        # Since we don't want to back-prop through the label_refinery network,
        # make the parameters of the teacher network not require gradients. This
        # saves some GPU memory.
        for param in self.label_refinery.parameters():
            param.requires_grad = False

    def forward(self, input):
        if self.training:
            refined_labels = self.label_refinery(input)
            refined_labels = F.softmax(refined_labels, dim=1)
            return refined_labels
        else:
            pass

class ModelWrapper(nn.Module):
    """Convenient wrapper class to train a model with a label refinery."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def LR_REGIME(self):
        # Training with label refinery does not change learing rate regime.
        # Return's wrapped model lr regime.
        return [1, 60, 0.002, 61, 120, 0.0002, 121, 200, 0.00002]
        # return [1, 30, 0.0002, 31, 120, 0.00002, 121, 200, 0.000002]

    def state_dict(self):
        return self.model.state_dict()

    def forward(self, input):
        if self.training:
            model_output = self.model(input)
            return model_output
        else:
            return self.model(input)