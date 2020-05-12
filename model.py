import sys
sys.path.insert(0, '/gpfs/software/Anaconda3/lib/python3.6/site-packages')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Log1PlusExp(autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        return x.where(torch.isinf(exp), exp.log1p())
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 + (-x).exp())
    

log_1_plus_exp = Log1PlusExp.apply

class RBM(nn.Module):
    r"""Restricted Boltzmann Machine.
    Args:
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_vis=784, n_target=2, n_hid=128, k=1):
        """Create a RBM."""
        super(RBM, self).__init__()
        self.n_vis = n_vis
        self.n_target = n_target
        self.n_hid = n_hid
        self.v = nn.Parameter(torch.randn(1, n_vis))
        self.y = nn.Parameter(torch.randn(1, n_target))
        self.h = nn.Parameter(torch.randn(1, n_hid))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))
        self.U = nn.Parameter(torch.randn(n_hid, n_target))
        self.k = k
        self.L = nn.Parameter(torch.randn(self.n_vis, self.n_vis))#.fill_diagonal_(0).triu()
    
#     def L_(self, L):    
# #         t = torch.randn(self.n_vis, self.n_vis)
#         mask = torch.eye(self.n_vis, self.n_vis).byte()
#         self.L.masked_fill_(mask, 0)
# #         self.L = nn.Parameter(self.L).to(device)
#         return nn.Parameter(self.L).to(device)

    def bernoulli(self, p):
        return F.relu(torch.sign(p - autograd.Variable(torch.rand(p.size()).to(device))))#.float()
    
    def visible_to_hidden(self, v, y):
        r"""Conditional sampling a hidden variable given a visible variable.
        Args:
            v (Tensor): The visible variable.
        Returns:
            Tensor: The hidden variable.
        """
#         print(y, y.is_cuda)
        p = torch.sigmoid(
            F.linear(v, self.W, self.h) + F.linear(y, self.U)
        )
        return self.bernoulli(p)

    def hidden_to_visible(self, h, v):
        r"""Conditional sampling a visible variable given a hidden variable.
        Args:
            h (Tendor): The hidden variable.
        Returns:
            Tensor: The visible variable.
        """
        L_ = torch.mm(self.L, self.L.t()).fill_diagonal_(0)
        p = torch.sigmoid(
            F.linear(h, self.W.t(), self.v) + F.linear(v, L_)
        )
        return self.bernoulli(p)
    
    def hidden_to_class(self, h):
        class_probablities = torch.exp(
            F.linear(h, self.U.t(), self.y)
        )
        class_probabilities = F.normalize(class_probablities, p = 1, dim = 1)
        labels = torch.argmax(class_probabilities, dim = 1)
        one_hot = torch.eye(self.n_target)
        return one_hot[labels].to(device)#.float()
    
    def forward(self, input_data):
        """Sampling the label given input data in time O(num_hidden * num_visible + num_classes * num_classes) for each example"""
        
    
        precomputed_factor = torch.matmul(input_data, self.W.t()) + self.h
        class_probabilities = torch.zeros((input_data.shape[0], self.n_target), device = input_data.device)#.to(device)

        for i in range(self.n_target):
            prod = torch.zeros(input_data.shape[0], device = input_data.device)
            prod += self.y.t()[i]
            for j in range(self.n_hid):
#                 prod += torch.log(1 + torch.exp(precomputed_factor[:,j] + self.U.t()[i, j]))
                prod += log_1_plus_exp(precomputed_factor[:,j] + self.U.t()[i, j])
            class_probabilities[:, i] = prod  

        copy_probabilities = torch.zeros(class_probabilities.shape, device = input_data.device)

        for c in range(self.n_target):
          for d in range(self.n_target):
            copy_probabilities[:, c] += torch.exp(-1*class_probabilities[:, c] + class_probabilities[:, d]).to(device = input_data.device)

        copy_probabilities = 1/copy_probabilities


        class_probabilities = copy_probabilities

        return class_probabilities

    def free_energy(self, v, y):
        r"""Free energy function.
        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}
        Args:
            v (Tensor): The visible variable.
        Returns:
            FloatTensor: The free energy value.
        """
        L_ = torch.mm(self.L, self.L.t()).fill_diagonal_(0)
        v_term = torch.matmul(v, self.v.t()) + v@L_@v.t() + torch.matmul(y, self.y.t())
        w_x_h = F.linear(v, self.W, self.h) + F.linear(y, self.U)
#         h_term = torch.sum(F.softplus(w_x_h), dim=1)
        
#         zr = autograd.Variable(torch.zeros(w_x_h.size())).to(device)
#         mask = torch.max(zr, w_x_h)
#         h_term = (((w_x_h - mask).exp() + (-mask).exp()).log() + (mask)).sum(1)

        h_term = log_1_plus_exp(w_x_h).sum()
        return torch.mean(-h_term - v_term)

    def gibb(self, v, y):
        r"""Compute the real and generated examples.
        Args:
            v (Tensor): The visible variable.
        Returns:
            (Tensor, Tensor): The real and generagted variables.
        """
        h = self.visible_to_hidden(v, y)
        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(h, v)
            y_gibb = self.hidden_to_class(h)
            h = self.visible_to_hidden(v_gibb, y_gibb)
        return v, v_gibb, y, y_gibb
    
class GRBM(RBM):

    '''
    Visisble layer can assume real values
    Hidden layer assumes Binarry Values only
    '''

    def hidden_to_visible(self, h, v):
        '''
        the visible units follow gaussian distributions here
        :params X: torch tensor shape = (n_samples , n_features)
        :returns X_prob - the new reconstructed layers(probabilities)
                sample_X_prob - sample of new layer(Gibbs Sampling)
        '''

        L_ = torch.mm(self.L, self.L.t()).fill_diagonal_(0)
        X_prob = F.linear(h, self.W.t(), self.v) + F.linear(v, L_)
#         print((X_prob != X_prob).any())
#         print(torch.isinf(X_prob).any())

        sample_X_prob = X_prob + torch.randn(X_prob.shape, device=device, requires_grad=True)
#         print((sample_X_prob != sample_X_prob).any())

        return self.bernoulli(sample_X_prob)