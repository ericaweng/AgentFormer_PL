
def get_tbd_split(sanity=False):
    all_files=['1_0.txt', '1_1.txt', '1_2.txt', '1_3.txt', '1_4.txt',
        '2_0.txt', '2_1.txt', '2_2.txt', '2_3.txt',
        '3_0.txt', '3_1.txt',
        '4_0.txt', '4_1.txt', '4_2.txt', '4_3.txt',
        '5_0.txt', '5_1.txt',
        '6_0.txt', '6_1.txt', '6_2.txt', '6_3.txt', '6_4.txt',
        '7_0.txt', '7_1.txt', '7_2.txt',
        '8_0.txt', '8_1.txt',
        '9_0.txt', '9_1.txt',
        '10_0.txt', '10_1.txt', '10_2.txt', '10_3.txt',
        '11_0.txt',
        '12_0.txt', '12_1.txt', '12_2.txt',
        '13_0.txt', '13_1.txt',
        '14_0.txt',
        '15_0.txt', '15_1.txt', '15_2.txt',
        '16_0.txt', '16_1.txt',
        '17_0.txt',
        '18_0.txt',
        '19_0.txt', '19_1.txt', '19_2.txt',
        '20_0.txt', '20_1.txt',
        '21_0.txt',
        '22_0.txt',
        '23_0.txt', '23_1.txt', '23_2.txt',
        '24_0.txt', '24_1.txt', '24_2.txt',
        '25_0.txt', '25_1.txt',
        '26_0.txt', '26_1.txt',
        '27_0.txt', '27_1.txt',
        '28_0.txt', '28_1.txt',
        '29_0.txt',
        '30_0.txt', '30_1.txt', '30_2.txt',
        '31_0.txt', '31_1.txt', '31_2.txt',
        '32_0.txt', '32_1.txt', '32_2.txt',
        '33_0.txt']
    if sanity:
        train = all_files[4:5]
        test = all_files[-2:-1]
    else:
        train = all_files[:60]
        test = all_files[60:]
    print(f"\n{len(train)=} {len(test)=}")
    return train, test, test


if __name__ == '__main__':
    train, test, _ = get_tbd_split()
    print(f"{train=}")
    print(f"{test=}")