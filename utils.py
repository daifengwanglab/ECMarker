import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/gpfs/software/Anaconda3/lib/python3.6/site-packages')
import torch
import torch.optim as optim
import torch.utils.data

def show_and_save(img, file_name):
    r"""Show and save the image.
    Args:
        img (Tensor): The image.
        file_name (Str): The destination.
    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)


def train(model, train_loader, opt=optim.Adam, n_epochs=20, lr=0.01, L1REG=0.01, L2REG=0.99, moment=None, device='cuda'):
    r"""Train a RBM model.
    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.
    Returns:
        The trained model.
    """
    # optimizer
    train_op = opt(model.parameters(), lr=lr, weight_decay=L2REG)
    if moment is not None:
        train_op = opt(model.parameters(), lr=lr, weight_decay=L2REG, momentum=moment)

    # train the RBM model
    model.train()

    for epoch in range(n_epochs):
        loss_ = []
        for _, (data, target) in enumerate(train_loader):
#             data = data.view(-1, 784).to(device)
#             target = torch.eye(10)[target]
#             target = target.to(device)
            v, v_gibbs, y, y_gibbs = model.gibb(data, target)
            loss = model.free_energy(v, y) - model.free_energy(v_gibbs, y_gibbs)
#             y_hat = model(data)
#             y_hat_class = torch.where(y_hat<0.5, torch.tensor(0.).to(device), torch.tensor(1.).to(device))
#             discr_loss = torch.nn.BCELoss()
#             loss += discr_loss(y_hat_class, target)

#             y_hat = model(data)
#             loss += -torch.mean(torch.log(y_hat))
            
            for param in model.parameters():
                loss += L1REG * torch.sum(torch.abs(param))
            
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()
            train_op.zero_grad()

        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))

    return model

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
            
def visualize_importances(feature_names, importances,
                          title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
#         print(feature_names[i], ": ", '%.3f'%(importances[i]))
        print(feature_names[i], ": ", (importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True, rotation='vertical')
        plt.xlabel(axis_title)
        plt.title(title)
#         plt.savefig("feature_importances.svg")

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.tensors[1][idx].cpu().numpy()[0] #for TensroDataset
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples