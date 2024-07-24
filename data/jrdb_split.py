
def get_jrdb_split_egomotion():
    TRAIN = ['clark-center-2019-02-28_0',
             'clark-center-2019-02-28_1',
             'clark-center-intersection-2019-02-28_0',
             'cubberly-auditorium-2019-04-22_0',  # small amount of rotation
             'forbes-cafe-2019-01-22_0',
             'gates-159-group-meeting-2019-04-03_0',
             'gates-to-clark-2019-02-28_1',  # linear movement
             'memorial-court-2019-03-16_0',
             'huang-2-2019-01-25_0',
             'huang-basement-2019-01-25_0',
             'meyer-green-2019-03-16_0',  # some rotation and movement
             'nvidia-aud-2019-04-18_0',  # small amount of rotation
             'packard-poster-session-2019-03-20_0',  # some rotation and movement
             'tressider-2019-04-26_2', ]
    # 14 train scenes with robot movement

    TEST = [
        'cubberly-auditorium-2019-04-22_1',
        'discovery-walk-2019-02-28_0',
        'discovery-walk-2019-02-28_1',
        # 'food-trucks-2019-02-12_0',
        'gates-ai-lab-2019-04-17_0',
        # 'gates-basement-elevators-2019-01-17_0',
        'gates-foyer-2019-01-17_0',
        'gates-to-clark-2019-02-28_0',
        # 'hewlett-class-2019-01-23_0',
        # 'hewlett-class-2019-01-23_1',
        'huang-2-2019-01-25_1',
        # 'huang-intersection-2019-01-22_0',
        'indoor-coupa-cafe-2019-02-06_0',
        # 'lomita-serra-intersection-2019-01-30_0',  #good
        'meyer-green-2019-03-16_1',
        'nvidia-aud-2019-01-25_0',
        # 'nvidia-aud-2019-04-18_1',
        # 'nvidia-aud-2019-04-18_2',
        'outdoor-coupa-cafe-2019-02-06_0',
        'quarry-road-2019-02-28_0',
        'serra-street-2019-01-30_0',
        'stlc-111-2019-04-19_1',
        'stlc-111-2019-04-19_2',
        # 'tressider-2019-03-16_2',
        'tressider-2019-04-26_0',
        # 'tressider-2019-04-26_1', # good
        'tressider-2019-04-26_3'
    ]
    # 27 total test scenes
    # 17 test scenes with robot movement
    return TRAIN, TEST, TEST


def get_jrdb_split_no_egomotion():
    TRAIN = ['bytes-cafe-2019-02-07_0',
               'gates-ai-lab-2019-02-08_0',
               'gates-basement-elevators-2019-01-17_1',
               'hewlett-packard-intersection-2019-01-24_0',
               'huang-lane-2019-02-12_0',
               'jordan-hall-2019-04-22_0',
               'packard-poster-session-2019-03-20_1',
               'packard-poster-session-2019-03-20_2',
               'stlc-111-2019-04-19_0',
               'svl-meeting-gates-2-2019-04-08_0',
               'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
               'tressider-2019-03-16_0',
               'tressider-2019-03-16_1', ]
    # 13 train scenes without robot movement

    TEST = [
        # 'cubberly-auditorium-2019-04-22_1',
        # 'discovery-walk-2019-02-28_0',
        # 'discovery-walk-2019-02-28_1',
        'food-trucks-2019-02-12_0',
        # 'gates-ai-lab-2019-04-17_0',
        'gates-basement-elevators-2019-01-17_0',
        # 'gates-foyer-2019-01-17_0',
        # 'gates-to-clark-2019-02-28_0',
        'hewlett-class-2019-01-23_0',
        'hewlett-class-2019-01-23_1',
        # 'huang-2-2019-01-25_1',
        'huang-intersection-2019-01-22_0',
        # 'indoor-coupa-cafe-2019-02-06_0',
        'lomita-serra-intersection-2019-01-30_0',  #good
        # 'meyer-green-2019-03-16_1',
        # 'nvidia-aud-2019-01-25_0',
        'nvidia-aud-2019-04-18_1',
        'nvidia-aud-2019-04-18_2',
        # 'outdoor-coupa-cafe-2019-02-06_0',
        # 'quarry-road-2019-02-28_0',
        # 'serra-street-2019-01-30_0',
        # 'stlc-111-2019-04-19_1',
        # 'stlc-111-2019-04-19_2',
        'tressider-2019-03-16_2',
        # 'tressider-2019-04-26_0',
        'tressider-2019-04-26_1', # good
        # 'tressider-2019-04-26_3'
    ]
    # 10 test scenes without robot movement
    return TRAIN, TEST, TEST


def get_jrdb_hst_split():  # the split used by HST paper
    TRAIN_SCENES = ['bytes-cafe-2019-02-07_0',
                    'clark-center-2019-02-28_0',
                    'clark-center-intersection-2019-02-28_0',
                    'cubberly-auditorium-2019-04-22_0',
                    'gates-159-group-meeting-2019-04-03_0',
                    'gates-ai-lab-2019-02-08_0',
                    'gates-to-clark-2019-02-28_1',
                    'hewlett-packard-intersection-2019-01-24_0',
                    'huang-basement-2019-01-25_0',
                    'huang-lane-2019-02-12_0',
                    'memorial-court-2019-03-16_0',
                    'meyer-green-2019-03-16_0',
                    'packard-poster-session-2019-03-20_0',
                    'packard-poster-session-2019-03-20_1',
                    'stlc-111-2019-04-19_0',
                    'svl-meeting-gates-2-2019-04-08_0',
                    'tressider-2019-03-16_0',
                    'tressider-2019-03-16_1',
                    'cubberly-auditorium-2019-04-22_1',
                    'discovery-walk-2019-02-28_0',
                    'food-trucks-2019-02-12_0',
                    'gates-ai-lab-2019-04-17_0',
                    'gates-foyer-2019-01-17_0',
                    'gates-to-clark-2019-02-28_0',
                    'hewlett-class-2019-01-23_1',
                    'huang-2-2019-01-25_1',
                    'indoor-coupa-cafe-2019-02-06_0',
                    'lomita-serra-intersection-2019-01-30_0',
                    'nvidia-aud-2019-01-25_0',
                    'nvidia-aud-2019-04-18_1',
                    'outdoor-coupa-cafe-2019-02-06_0',
                    'quarry-road-2019-02-28_0',
                    'stlc-111-2019-04-19_1',
                    'stlc-111-2019-04-19_2',
                    'tressider-2019-04-26_0',
                    'tressider-2019-04-26_1']

    TEST = ['clark-center-2019-02-28_1',
            'forbes-cafe-2019-01-22_0',
            'gates-basement-elevators-2019-01-17_1',
            'huang-2-2019-01-25_0',
            'jordan-hall-2019-04-22_0',
            'nvidia-aud-2019-04-18_0',
            'packard-poster-session-2019-03-20_2',
            'svl-meeting-gates-2-2019-04-08_1',
            'tressider-2019-04-26_2',
            'discovery-walk-2019-02-28_1',
            'gates-basement-elevators-2019-01-17_0',
            'hewlett-class-2019-01-23_0',
            'huang-intersection-2019-01-22_0',
            'meyer-green-2019-03-16_1',
            'nvidia-aud-2019-04-18_2',
            'serra-street-2019-01-30_0',
            'tressider-2019-03-16_2',
            'tressider-2019-04-26_3']
    # number of scenes: 18

    return TRAIN_SCENES, TEST, TEST


def get_jrdb_split_full():  # the split as established by the JRDB dataset authors
    TRAIN = [
        'bytes-cafe-2019-02-07_0',
        'clark-center-2019-02-28_0',
        'clark-center-2019-02-28_1',
        'clark-center-intersection-2019-02-28_0',
        'cubberly-auditorium-2019-04-22_0',
        'forbes-cafe-2019-01-22_0',
        'gates-159-group-meeting-2019-04-03_0',
        'gates-ai-lab-2019-02-08_0',
        'gates-basement-elevators-2019-01-17_1',
        'gates-to-clark-2019-02-28_1',
        'hewlett-packard-intersection-2019-01-24_0',
        'huang-2-2019-01-25_0',
        'huang-basement-2019-01-25_0',
        'huang-lane-2019-02-12_0',
        'jordan-hall-2019-04-22_0',
        'memorial-court-2019-03-16_0',
        'meyer-green-2019-03-16_0',
        'nvidia-aud-2019-04-18_0',
        'packard-poster-session-2019-03-20_0',
        'packard-poster-session-2019-03-20_1',
        'packard-poster-session-2019-03-20_2',
        'stlc-111-2019-04-19_0',
        'svl-meeting-gates-2-2019-04-08_0',
        'svl-meeting-gates-2-2019-04-08_1',
        'tressider-2019-03-16_0',
        'tressider-2019-03-16_1',
        'tressider-2019-04-26_2'
    ]
    # number of scenes: 27

    TEST = [
        'cubberly-auditorium-2019-04-22_1',
        'discovery-walk-2019-02-28_0',
        'discovery-walk-2019-02-28_1',
        'food-trucks-2019-02-12_0',
        'gates-ai-lab-2019-04-17_0',
        'gates-basement-elevators-2019-01-17_0',
        'gates-foyer-2019-01-17_0',
        'gates-to-clark-2019-02-28_0',
        'hewlett-class-2019-01-23_0',
        'hewlett-class-2019-01-23_1',
        'huang-2-2019-01-25_1',
        'huang-intersection-2019-01-22_0',
        'indoor-coupa-cafe-2019-02-06_0',
        'lomita-serra-intersection-2019-01-30_0',
        'meyer-green-2019-03-16_1',
        'nvidia-aud-2019-01-25_0',
        'nvidia-aud-2019-04-18_1',
        'nvidia-aud-2019-04-18_2',
        'outdoor-coupa-cafe-2019-02-06_0',
        'quarry-road-2019-02-28_0',
        'serra-street-2019-01-30_0',
        'stlc-111-2019-04-19_1',
        'stlc-111-2019-04-19_2',
        'tressider-2019-03-16_2',
        'tressider-2019-04-26_0',
        'tressider-2019-04-26_1',
        'tressider-2019-04-26_3'
    ]
    # number of scenes: 27
    return TRAIN, TEST, TEST


def get_jackrabbot_split_half_and_half_tiny():
    train = ['gates-basement-elevators-2019-01-17_1',
             'huang-lane-2019-02-12_0',
             'stlc-111-2019-04-19_0', ]

    val = ['gates-basement-elevators-2019-01-17_1',
           'huang-lane-2019-02-12_0',
           'stlc-111-2019-04-19_0', ]

    test = ['gates-basement-elevators-2019-01-17_1',
            'huang-lane-2019-02-12_0',
            'stlc-111-2019-04-19_0',]
    return train, val, test


def get_jackrabbot_split_half_and_half():  # for each scene, use half for training and half for testing
    train = ['bytes-cafe-2019-02-07_0',
             'gates-ai-lab-2019-02-08_0',
             'gates-basement-elevators-2019-01-17_1',
             'hewlett-packard-intersection-2019-01-24_0',
             'huang-lane-2019-02-12_0',
             'jordan-hall-2019-04-22_0',
             'packard-poster-session-2019-03-20_1',
             'packard-poster-session-2019-03-20_2',
             'stlc-111-2019-04-19_0',
             'svl-meeting-gates-2-2019-04-08_0',
             'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
             'tressider-2019-03-16_0',
             'tressider-2019-03-16_1']
    val = ['bytes-cafe-2019-02-07_0',
             'gates-ai-lab-2019-02-08_0',
             'gates-basement-elevators-2019-01-17_1',
             'hewlett-packard-intersection-2019-01-24_0',
             'huang-lane-2019-02-12_0',
             'jordan-hall-2019-04-22_0',
             'packard-poster-session-2019-03-20_1',
             'packard-poster-session-2019-03-20_2',
             'stlc-111-2019-04-19_0',
             'svl-meeting-gates-2-2019-04-08_0',
             'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
             'tressider-2019-03-16_0',
             'tressider-2019-03-16_1']
    test = ['bytes-cafe-2019-02-07_0',
             'gates-ai-lab-2019-02-08_0',
             'gates-basement-elevators-2019-01-17_1',
             'hewlett-packard-intersection-2019-01-24_0',
             'huang-lane-2019-02-12_0',
             'jordan-hall-2019-04-22_0',
             'packard-poster-session-2019-03-20_1',
             'packard-poster-session-2019-03-20_2',
             'stlc-111-2019-04-19_0',
             'svl-meeting-gates-2-2019-04-08_0',
             'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
             'tressider-2019-03-16_0',
             'tressider-2019-03-16_1']

    return train, val, test


def get_jackrabbot_split_sanity():
    """"""
    train = ['huang-basement-2019-01-25_0',]
    val = ['huang-basement-2019-01-25_0',]
    test = ['huang-basement-2019-01-25_0',]

    return train, val, test


def get_jrdb_training_split_erica():  # JRDB TRAINING DATA ONLY: all have labels
    train = ['bytes-cafe-2019-02-07_0',
             'gates-ai-lab-2019-02-08_0',
             'gates-basement-elevators-2019-01-17_1',
             'hewlett-packard-intersection-2019-01-24_0',
             'huang-lane-2019-02-12_0',
             'jordan-hall-2019-04-22_0',
             'packard-poster-session-2019-03-20_1',
             'packard-poster-session-2019-03-20_2']

    val = ['stlc-111-2019-04-19_0',
           'svl-meeting-gates-2-2019-04-08_0',
           'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
           'tressider-2019-03-16_0',
           'tressider-2019-03-16_1']

    test = ['stlc-111-2019-04-19_0',
            'svl-meeting-gates-2-2019-04-08_0',
            'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
            'tressider-2019-03-16_0',
            'tressider-2019-03-16_1']

    return train, val, test
