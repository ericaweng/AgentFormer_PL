
def get_jackrabbot_split():
    train = ['bytes-cafe-2019-02-07_0',
             'gates-ai-lab-2019-02-08_0',
             'gates-basement-elevators-2019-01-17_1',
             'hewlett-packard-intersection-2019-01-24_0',
             'huang-lane-2019-02-12_0',
             'jordan-hall-2019-04-22_0',
             'packard-poster-session-2019-03-20_1',
             'packard-poster-session-2019-03-20_2']

    egomotion_adjusted_data = [
            'huang-basement-2019-01-25_0',
            'nvidia-aud-2019-04-18_0',
            'clark-center-2019-02-28_0'
    ]

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


    #### Ronnie's split #####
    # train = [
    #          'gates-ai-lab-2019-02-08_0',
    #          'hewlett-packard-intersection-2019-01-24_0',
    #          'huang-lane-2019-02-12_0',
    #          'jordan-hall-2019-04-22_0',
    #          'packard-poster-session-2019-03-20_1',
    #          'packard-poster-session-2019-03-20_2',
    #          'stlc-111-2019-04-19_0',
    #          'svl-meeting-gates-2-2019-04-08_0',
    #          'svl-meeting-gates-2-2019-04-08_1',
    #          # 'tressider-2019-03-16_1',  # ronnie missing this scene
    # ]
    #
    # val = [
    #         'bytes-cafe-2019-02-07_0',
    #         'tressider-2019-03-16_0',
    #         'gates-basement-elevators-2019-01-17_1'
    # ]
    #
    # test = [
    #         'bytes-cafe-2019-02-07_0',
    #         'tressider-2019-03-16_0',
    #         'gates-basement-elevators-2019-01-17_1'
    # ]

    return train, val, test
