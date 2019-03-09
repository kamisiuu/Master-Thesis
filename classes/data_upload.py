def trainupload( path, header):
    train = pd.read_csv(path, header=header, index_col=None)
    return train


def testupload( path, header):
    test = pd.read_csv(path, header=header, index_col=None)
    return test