def calc_reward(array_actions, n_members):

    # if array_actions[0] == 0:
    #     reward = 0
    # elif array_actions[0] == 1:
    #     reward = -1 * 50
    # elif array_actions[0] == 2:
    #     reward = -1 * (50 + n_members * 9)
    # elif array_actions[0] == 3:
    #     reward = -1 * (100 + n_members * 9)
    # elif array_actions[0] == 4:
    #     reward = -1 * (200 + n_members * 9)
    # elif array_actions[0] == 5:
    #     reward = -1 * (200 + n_members * 18)
    # elif array_actions[0] == 6:
    #     reward = -1 * (300 + n_members * 18)
    # elif array_actions[0] == 7:
    #     reward = -1 * (300 + n_members * 36)
    # elif array_actions[0] == 8:
    #     reward = -1 * (400 + n_members * 36)
    # elif array_actions[0] == 9:
    #     reward = -1 * (500 + n_members * 36 + n_members * 199)

    if array_actions[0] == 999:
        reward = -1 * 100000
        penalty = 1 * (500 + 36 * n_members + n_members * 398)
    elif array_actions[0] == 0:
        reward = 10000
        penalty = 0
    elif array_actions[0] == 1:
        reward = 500
        penalty = 1 * 50
    elif array_actions[0] == 2:
        reward = 250
        penalty = 1 * (50 + n_members * 9)
    elif array_actions[0] == 3:
        reward = 125
        penalty = 1 * (100 + n_members * 9)
    elif array_actions[0] == 4:
        reward = 62
        penalty = 1 * (200 + n_members * 9)
    elif array_actions[0] == 5:
        reward = 31
        penalty = 1 * (200 + n_members * 18)
    elif array_actions[0] == 6:
        reward = 16
        penalty = 1 * (300 + n_members * 18)
    elif array_actions[0] == 7:
        reward = 8
        penalty = 1 * (300 + n_members * 36)
    elif array_actions[0] == 8:
        reward = 4
        penalty = 1 * (400 + n_members * 36)
    elif array_actions[0] == 9:
        reward = 2
        penalty = (500 + n_members * 36 + n_members * 199)

    return -1*penalty, penalty
