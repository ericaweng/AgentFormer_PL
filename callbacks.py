from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointCustom(ModelCheckpoint):
    def __init__(self, visualize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_train_this_time = False
        self.visualize = visualize

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.log_train_this_time:
            pl_module.args.save_num = 5
            pl_module.log_viz(pl_module.outputs, 'train')
            pl_module.outputs = None
            self.log_train_this_time = False


    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.visualize and (self.best_model_score and trainer.callback_metrics.get(self.monitor)
                               or pl_module.current_epoch == 0
                               and hasattr(pl_module, 'outputs') and len(pl_module.outputs) > 0):
            if pl_module.current_epoch == 0:
                pl_module.args.save_num = 5
                pl_module.log_viz(pl_module.outputs, 'val')
                pl_module.outputs = None
                self.log_train_this_time = True
            elif trainer.callback_metrics.get(self.monitor) <= self.best_model_score:
                print("This is a new best model!")
                pl_module.args.save_num = 12
                pl_module.log_viz(pl_module.outputs, 'val')
                pl_module.outputs = None
                self.log_train_this_time = True
        elif pl_module.args.save_viz_every_time and hasattr(pl_module, 'outputs') and len(pl_module.outputs) > 0:
            pl_module.log_viz(pl_module.outputs, 'val')
            pl_module.outputs = None
            self.log_train_this_time = True
