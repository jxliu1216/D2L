from torch.utils import data

def load_array(data_arrays, batch_size, is_train=True):
    """ Construct a Pytorch data iter """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)