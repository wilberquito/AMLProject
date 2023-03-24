import torch

# reference (https://gist.github.com/wassname/c15a2b72df716f2fa1299661c1414e6b)
class AdamStepLR(torch.optim.Adam):
    """Combine Adam and lr_scheduler.StepLR so we can use it as a normal optimiser"""
    def __init__(self,
                 params,
                 lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 weight_decay=0,
                 step_size=50000,
                 gamma=0.5,
                 verbose=False):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self,
                                                         step_size=step_size,
                                                         gamma=gamma,
                                                         verbose=verbose)

    def step(self):
        self.scheduler.step()
        return super().step()