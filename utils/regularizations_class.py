import torch

class Regularization:
    """
    A class for computing a regularization term.

    Currently, only LP regularization is implemented, which is computed as:

        output_loss = reg_const * (Σ |param_i|^p)^(1/p)

    where the sum is over all parameters of the model that require gradients.

    Attributes
    ----------
    reg_type : str
        The type of regularization to use. (Currently supported: 'LP')
    reg_const : float
        The regularization constant (λ).
    p : int or float
        The exponent for the LP regularization (e.g., 1 for L1, 2 for L2, etc.).
    """
    
    def __init__(self, reg_type='LP', reg_const=1e-5, p=2):
        """
        Initialize the regularization.

        Parameters
        ----------
        reg_type : str, optional
            The type of regularization to apply (default is 'LP').
        reg_const : float, optional
            The regularization constant (λ) (default is 1e-5).
        p : int or float, optional
            The exponent for the LP regularization (e.g., 1 for L1, 2 for L2, etc.) (default is 2).
        """
        self.reg_type = reg_type
        self.reg_const = reg_const
        self.p = p

    def forward(self, model):
        """
        Compute the regularization term for the specified model.

        This method iterates over all parameters of the model (only those with `requires_grad=True`)
        and, if the regularization type is 'LP', computes the term as:

            output_loss = reg_const * (Σ |param|^p)^(1/p)

        Parameters
        ----------
        model : torch.nn.Module
            The model for which to compute the regularization term.

        Returns
        -------
        output_loss : torch.Tensor
            The computed regularization term.

        Notes
        -----
        If `reg_type` is not 'LP', an error message is printed.
        """
        if self.reg_type == 'LP':
            regularization_term = 0
            for param in model.parameters():
                if param.requires_grad:
                    regularization_term += torch.sum(torch.abs(param) ** self.p)
             
            output_loss = self.reg_const * regularization_term**(1/self.p)
        else:
            print("Error: unsupported regularization type.")
            
        return output_loss
