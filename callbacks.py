from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointCustom(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_train_this_time = False

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if self.log_train_this_time:
            pl_module.args.save_num = 2
            pl_module.log_viz(pl_module.outputs, 'train')
            self.log_train_this_time = False


    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.best_model_score:
            current_score = trainer.callback_metrics.get("val/ADE_joint")
            if current_score <= self.best_model_score:
                print("This is a new best model!")
                pl_module.args.save_num = 10
                pl_module.log_viz(pl_module.outputs, 'val')
                self.log_train_this_time = True
        elif pl_module.args.save_viz_every_time:
            pl_module.log_viz(pl_module.outputs, 'val')
            self.log_train_this_time = True
