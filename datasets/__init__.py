from scipy import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from urllib.request import urlretrieve


_urls = {
    'indian_pines': {
        'image': 'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'labels': 'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
    },
    'pavia_university': {
        'image': 'https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'labels': 'https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'
    },
    'pavia_center': {
        'image': 'https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
        'labels': 'https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'
    },
    'salinas': {
        'image': 'https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        'labels': 'https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat'
    },
    'salinas_a': {
        'image': 'https://www.ehu.eus/ccwintco/uploads/1/1a/SalinasA_corrected.mat',
        'labels': 'https://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat'
    },
    'botswana': {
        'image': 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        'labels': 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
    },
}


def _download(origin, filepath):
    try:
        print(f'downloading from "{origin}"...')
        local_path, _ = urlretrieve(origin, filepath)
        print(f'file downloaded successfully')
    except:
        raise Exception('download failed')


def load_data(name, show_info=True):
    if name not in _urls.keys():
        raise ValueError(f'dataset {name} does not exist')
    # create dir
    path = Path(__file__).resolve().parent.joinpath(name)
    path.mkdir(parents=True, exist_ok=True)
    # download if not done already
    if not path.joinpath('image.mat').is_file():
        _download(_urls[name]['image'], path.joinpath('image.mat'))
    if not path.joinpath('labels.mat').is_file():
        _download(_urls[name]['labels'], path.joinpath('labels.mat'))
    # load data
    image = io.loadmat(path.joinpath('image.mat'))
    labels = io.loadmat(path.joinpath('labels.mat'))
    X = image[list(image.keys())[-1]]
    y = labels[list(labels.keys())[-1]]
    # show info
    if show_info:
        print('Dataset info:')
        print(f'  - Scene name: {name}')
        print(f'  - Height: {X.shape[0]}')
        print(f'  - Width: {X.shape[1]}')
        print(f'  - Number of bands: {X.shape[2]}')
        print(f'  - Number of labels: {np.unique(y).shape[0]}')
    return X, y


