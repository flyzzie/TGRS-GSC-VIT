from .cnn2d import cnn2d
from .sprn import SPRN
from .cnn3d import cnn3d
from .hybridsn import hybridsn
from .spectralformer import spectralformer
from .ssftt import ssftt
from .gaht import gaht
from .gscvit import gscvit
from .morphFormer import morphFormer
from .caevt import caevt
def get_model(model_name, dataset_name, patch_size):
    if model_name == 'cnn2d':
        model = cnn2d(dataset=dataset_name)

    elif model_name == 'sprn':
        model = SPRN(dataset=dataset_name)

    elif model_name == 'cnn3d':
        model = cnn3d(dataset_name, patch_size)

    elif model_name == 'hybridsn':
        model = hybridsn(dataset_name, patch_size)

    elif model_name == 'spectralformer':
        model = spectralformer(dataset_name, patch_size)

    elif model_name == 'ssftt':
        model = ssftt(dataset_name, patch_size)

    elif model_name == 'gaht':
        model = gaht(dataset_name, patch_size)

    elif model_name == 'morphFormer':
        model = morphFormer(16, 80, 10, False, 8)

    elif model_name == 'gscvit':
        model = gscvit(dataset_name)

    elif model_name == 'caevt':
        model = caevt(dataset_name)

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model

