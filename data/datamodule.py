import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from data.dataset import AgentFormerDataset


class AgentFormerDataModule(pl.LightningDataModule):
    def __init__(self, cfg, args):#batch_size):#
        super().__init__()
        self.cfg = cfg
        # self.batch_size = batch_size
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def get_dataloader(self, mode):
        phase = 'testing' if 'val' in mode or 'test' in mode else 'training'
        trial_ds_size = self.args.trial_ds_size if self.args.trial else None
        randomize_trial_data = self.args.randomize_trial_data if self.args.trial else False
        ds = AgentFormerDataset(self.cfg, split=mode, phase=phase, trial_ds_size=trial_ds_size,
                                randomize_trial_data=randomize_trial_data,
                                frames_list=self.args.frames_list, start_frame=self.args.start_frame, args=self.args)
        shuffle = False if ('val' in mode or 'test' in mode or self.args.trial
                            or self.args.mode == 'check_dl' or self.args.mode == 'viz') else True
        dataloader = DataLoader(ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                pin_memory=True, collate_fn=ds.collate, shuffle=shuffle, drop_last=shuffle)
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader(self.args.test_dataset if self.args.trial else "train")

    def val_dataloader(self):
        return self.get_dataloader(self.args.test_dataset if self.args.trial else "val")

    def test_dataloader(self):
        return self.get_dataloader(self.args.test_dataset if self.args.trial else "test")
