import torch

def accuracy(an, labels):
    """Return the accuracy of the prediction according to the true labels.
    The input should be in the shape of (n_samples, tensor) and from the last unit of the network."""
    top_p, top_class = an.topk(1, dim=1) # top_p is the highest value in each sample, top_class is index of the highest value
    equals = top_class == labels.view(*top_class.shape) # usually they have the same n_samples, but different in the second dimension
    accuracy_score = torch.mean(equals.type(torch.FloatTensor))
    return accuracy_score