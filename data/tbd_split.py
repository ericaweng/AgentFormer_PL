def get_tbd_split_small():
    train=['1_0','1_1','1_2','1_3','1_4']
    test=['10_0']# '2_0']#
    return train, test, test


def get_tbd_split_sanity2():
    train=['7_1']
    test=['1_0']
    return train, test, test


def get_tbd_split_allan():
    
    all_files = ['1_0', '1_1', '1_2', '1_3', '1_4',
                 '2_0', '2_1', '2_2', '2_3',
                 '3_0', '3_1',
                 '4_0', '4_1', '4_2', '4_3',
                 '5_0', '5_1',
                 '6_0', '6_1', '6_2', '6_3', '6_4',
                 '7_0', '7_1', '7_2',
                 '8_0', '8_1',
                 '9_0', '9_1',
                 '10_0', '10_1', '10_2', '10_3',
                 '11_0',
                 '12_0', '12_1', '12_2',
                 '13_0', '13_1',
                 '14_0',
                 '15_0', '15_1', '15_2',
                 '16_0', '16_1',
                 '17_0',
                 '18_0',
                 '19_0', '19_1', '19_2',
                 '20_0', '20_1',
                 '21_0',
                 '22_0',
                 '23_0', '23_1', '23_2',
                 '24_0', '24_1', '24_2',
                 '25_0', '25_1',
                 '26_0', '26_1',
                 '27_0', '27_1',
                 '28_0', '28_1',
                 '29_0',
                 '32_0', '32_1', '32_2',
                 '33_0']
    val = ['30_0', '30_1', '30_2',
           '31_0', '31_1', '31_2']

    return all_files, val, val


def get_tbd_interesting_scenes_and_frames():
    from traj_toolkit.data_utils.tbd_interesting_scenes import INTERESTING_SCENES
    sorted_scenes = dict(sorted(INTERESTING_SCENES.items(), key=lambda x: x[0]))
    return sorted_scenes


def get_frame_ids_with_margin(scenes_dict, frame_skip=4, margin=2):
    new_dict = {}
    
    for scene, frames_dict in scenes_dict.items():
        frame_ids = []
        for frame in frames_dict.keys():
            # Create a range of frames before and after the current frame
            frame_range = list(range(frame - frame_skip * margin, frame + frame_skip * margin + 1, frame_skip))
            # Add this range to the list of frame_ids, ensuring no duplicates
            frame_ids.extend(frame_range)
        
        # Remove any duplicates and sort the frame_ids
        new_dict[scene] = sorted(set(frame_ids))
    
    return new_dict

def get_test_tbd_interesting_scenes():
    from traj_toolkit.data_utils.tbd_interesting_scenes import INTERESTING_SCENES
    sorted_scenes = INTERESTING_SCENES
    scenes = [k for k, v in sorted_scenes.items() if len(v) > 0]
    lenn = int(len(scenes)*3/4)
    return scenes[:lenn], scenes, scenes


def get_tbd_interesting_scenes():
    from traj_toolkit.data_utils.tbd_interesting_scenes import INTERESTING_SCENES
    sorted_scenes = INTERESTING_SCENES#get_tbd_interesting_scenes_and_frames()
    scenes = [k for k, v in sorted_scenes.items() if len(v) > 0]
    lenn = int(len(scenes)*3/4)
    return  scenes[:lenn], scenes[lenn:],scenes[lenn:]


def get_tbd_split(sanity=False):
    all_files=['1_0', '1_1', '1_2', '1_3', '1_4',
        '2_0', '2_1', '2_2', '2_3',
        '3_0', '3_1',
        '4_0', '4_1', '4_2', '4_3',
        '5_0', '5_1',
        '6_0', '6_1', '6_2', '6_3', '6_4',
        '7_0', '7_1', '7_2',
        '8_0', '8_1',
        '9_0', '9_1',
        '10_0', '10_1', '10_2', '10_3',
        '11_0',
        '12_0', '12_1', '12_2',
        '13_0', '13_1',
        '14_0',
        '15_0', '15_1', '15_2',
        # ]
    # val=[
        '16_0', '16_1',
        '17_0',
        '18_0',
        '19_0', '19_1', '19_2',
        '20_0', '20_1',
        '21_0',
        '22_0',
        '23_0', '23_1', '23_2',
        '24_0', '24_1', '24_2',
        # ]
    # test = [
        '25_0', '25_1',
        '26_0', '26_1',
        '27_0', '27_1',
        '28_0', '28_1',
        '29_0',
        '30_0', '30_1', '30_2',
        '31_0', '31_1', '31_2',
        '32_0', '32_1', '32_2',
        '33_0']
    if sanity:
        train = all_files[4:5]
        test = all_files[-2:-1]
    else:
        train = all_files[:60]
        test = all_files[60:]
    # print(f"\ntbd split: {len(train)=} {len(test)=}")
    # return all_files, val, test
    return train, test, test


if __name__ == '__main__':
    train, test, _ = get_tbd_split()
    print(f"{train=}")
    print(f"{test=}")