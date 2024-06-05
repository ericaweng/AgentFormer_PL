"""generate configs for new experiments"""
import os
from pathlib import Path


def save_new_cfg(filename, cfg):
    print("cfg:", cfg)
    print(f"save to {filename}?")
    with open(filename, 'w') as f:
        f.write(cfg)

def main():
    cfgs_path = 'cfg'
    total_new_cfgs = 0

    new_dir = f'{cfgs_path}/jrdb/PiFeNet'
    os.makedirs(new_dir, exist_ok=True)

    for file in Path(f'{cfgs_path}/jrdb/ss3d_mot/').rglob('*.yml'):
        with open(file, 'r') as f:
            cfg = f.read()
        save_prefix = str(file).replace('ss3d_mot', 'PiFeNet')
        save_prefix = str(save_prefix).replace('ss3d', 'pife')
        # print(f"{save_prefix=}")
        cfg = cfg.replace('ss3d_mot', 'PiFeNet')
        dest_filename = save_prefix
        # print(cfg)
        if not os.path.exists(os.path.dirname(dest_filename)):
            os.makedirs(os.path.dirname(dest_filename))
        save_new_cfg(dest_filename, cfg)

        # ----------------------------------------------------
        total_new_cfgs += 1
        continue

    print(f"total new cfgs: {total_new_cfgs}")


if __name__ == '__main__':
    main()
