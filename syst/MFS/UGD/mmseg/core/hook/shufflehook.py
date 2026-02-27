from mmcv.runner import Hook

class IterShuffleHook(Hook):
    def __init__(self, num_iters=1000,dataset=None):
        self.num_iters = num_iters
        self.dataset=dataset
    
    def before_train_iter(self, runner):
        if runner.iter % self.num_iters == 0:
            if hasattr(self.dataset, 'shuffle'):
                self.dataset.shuffle()
                print(f"Dataset shuffled at iteration {runner.iter}")
                
class EpochShuffleHook(Hook):
    def __init__(self, dataset=None):
        self.dataset = dataset
    
    def before_train_epoch(self, runner):
        if hasattr(self.dataset, 'shuffle'):
            self.dataset.shuffle()
            print(f"Dataset shuffled at epoch {runner.epoch}")
