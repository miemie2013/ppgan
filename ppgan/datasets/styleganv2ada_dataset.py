# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import PIL
import json
from .base_dataset import BaseDataset
from .builder import DATASETS
import os
import zipfile
import numpy as np
from PIL import Image

try:
    import pyspng
except ImportError:
    pyspng = None



@DATASETS.register()
class StyleGANv2ADADataset(BaseDataset):
    """
    """
    def __init__(self, dataroot, is_train, preprocess, resolution=None,
                 max_size=None, use_labels=False, xflip=False, random_seed=0, len_phases=4):
        """Initialize single dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.
        """
        super(StyleGANv2ADADataset, self).__init__(preprocess)

        self.dataroot = dataroot
        self.is_train = is_train
        self.len_phases = len_phases

        self._type = 'dir'
        self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.dataroot) for root, _dirs, files in os.walk(self.dataroot) for fname in files}

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self.dataroot))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        # 父类
        # super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.dataroot)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self.dataroot, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        image_gen_c = [self.get_label(np.random.randint(len(self))) for _ in range(self.len_phases)]
        return image.copy(), self.get_label(idx), image_gen_c, self._raw_idx[idx]

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])


    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    def __len__(self):
        size = self._raw_idx.size
        return size

    def prepare_data_infos(self, dataroot):
        pass



@DATASETS.register()
class StyleGANv2ADATestDataset(BaseDataset):
    """
    """
    def __init__(self, seeds, z_dim, preprocess=None):
        """Initialize single dataset class.

        Args:
            seeds (list[int]): seeds.
            z_dim (int): z_dim.
        """
        super(StyleGANv2ADATestDataset, self).__init__(preprocess)
        self.seeds = seeds
        self.z_dim = z_dim

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        z = np.random.RandomState(seed).randn(self.z_dim, )
        datas = {
            'z': z,
            'seed': seed,
        }
        return datas

    def __len__(self):
        size = len(self.seeds)
        return size

    def prepare_data_infos(self, dataroot):
        pass


