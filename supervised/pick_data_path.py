def pick_data_path(data_name,dim):
    if data_name == 'greedy':
        if dim == 20:
            data_path = 'data/data_greedy_20.csv'
        elif dim == 10:
            data_path = 'data/data_greedy_10.csv'
    elif data_name == 'a_star':
        if dim == 20:
            data_path = 'data/data_astar_20.csv'
        elif dim == 10:
            data_path = 'data/data_astar_10.csv'
    return data_path