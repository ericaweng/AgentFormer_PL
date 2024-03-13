import yaml
import os
import os.path as osp
import glob
from easydict import EasyDict


class Config:

    def __init__(self, cfg_id):
        self.id = cfg_id
        cfg_path = 'cfg/**/%s.yml' % cfg_id
        files = glob.glob(cfg_path, recursive=True)
        if len(files) == 0:
            raise ValueError('No config file found for cfg_id %s' % cfg_id)

        assert (len(files) == 1), files
        self.yml_dict = EasyDict(yaml.safe_load(open(files[0], 'r')))

        self.cfg_path = os.path.expanduser(files[0])

    def get_last_epoch(self):
        model_files = sorted(glob.glob(os.path.join(self.model_dir, 'model_*.p')))
        if len(model_files) == 0:
            return None
        else:
            model_file = osp.basename(model_files[-1])
            epoch = int(osp.splitext(model_file)[0].split('model_')[-1])
            return epoch            

    def __getattribute__(self, name):
        yml_dict = super().__getattribute__('yml_dict')
        if name in yml_dict:
            return yml_dict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            yml_dict = super().__getattribute__('yml_dict')
        except AttributeError:
            return super().__setattr__(name, value)
        if name in yml_dict:
            yml_dict[name] = value
        else:
            return super().__setattr__(name, value)

    def get(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return default
