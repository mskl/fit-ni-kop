def load_bag_data(x_file, y_file):
    with open(x_file) as x, open(y_file) as y:
        return tuple(zip(x.readlines(), y.readlines()))
