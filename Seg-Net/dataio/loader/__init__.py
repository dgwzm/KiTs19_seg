from List_dataset import LITS_dataset

def get_dataset(name):
    """get_dataset
    :param name:
    """
    return {

        'List': LITS_dataset,
    }[name]

def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
