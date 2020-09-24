import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class LfwDataset:
    def __init__(self, shuffle_count=1000):
        self.data_splits = {}
        self.data_sample = None
        self.base_img_shape = None
        self.shuffle_count = shuffle_count
        
    def get_base_img_shape(self):
        return self.base_img_shape
    
    @staticmethod
    def scale_and_crop_img(ds_iter):
        return (tf.cast(ds_iter['image'][45:205, 45:205, :], dtype='float32') / 128.0) - 1.0

    @staticmethod
    def unscale_img(img):
        return np.clip(tf.squeeze(img / 2.0 + 0.5), 0.0, 1.0)
    
    @staticmethod
    def resample(img, scale):
        if abs(scale) == 1:
            return img
        
        if scale < 0:
            new_side = int(img.shape[-3] // abs(scale))
        else:
            new_side = int(img.shape[-3] * scale)
            
        return tf.image.resize(img, [new_side, new_side])
        
    def tfds_load(self, scale=1, load_count=None):
        if load_count is None:
            # This loads everything by percentages of the whole dataset
            train_ds_unbatched, train_info = tfds.load('lfw', split='train[:80%]', with_info=True)
            val_ds_unbatched, val_info = tfds.load('lfw', split='train[80%:90%]', with_info=True)
            test_ds_unbatched, test_info = tfds.load('lfw', split='train[90%:]', with_info=True)
        else:
            # For experimentation we might load just a subset
            train_ct = int(load_count * 0.8)
            val_ct = int(load_count * 0.1) + train_ct
            test_ct = int(load_count * 0.1) + val_ct
            tr_split = f'train[:{train_ct}]'
            vl_split = f'train[{train_ct}:{val_ct}]'
            ts_split = f'train[{val_ct}:{test_ct}]'
            train_ds_unbatched, train_info = tfds.load('lfw', split=tr_split, with_info=True)
            val_ds_unbatched, val_info = tfds.load('lfw', split=vl_split, with_info=True)
            test_ds_unbatched, test_info = tfds.load('lfw', split=ts_split, with_info=True)
            
        sizes = []

        shuf = self.shuffle_count
        train_ds_unbatched = train_ds_unbatched.map(LfwDataset.scale_and_crop_img).shuffle(shuf)
        val_ds_unbatched = val_ds_unbatched.map(LfwDataset.scale_and_crop_img)
        test_ds_unbatched = test_ds_unbatched.map(LfwDataset.scale_and_crop_img)
        
        
        rescale_func = lambda img: LfwDataset.resample(img, scale)
        train_ds_unbatched = train_ds_unbatched.map(rescale_func)
        val_ds_unbatched = val_ds_unbatched.map(rescale_func)
        test_ds_unbatched = test_ds_unbatched.map(rescale_func)

        samples = []
        for ds in [train_ds_unbatched, val_ds_unbatched, test_ds_unbatched]:
            for i,batch in enumerate(ds):
                if i < 3:
                    samples.append(batch)
                pass
            sizes.append(i+1)

        print('Train samples:', sizes[0])
        print('Val samples:  ', sizes[1])
        print('Test samples: ', sizes[2])

        test_img_sample = []
        for i,img in enumerate(test_ds_unbatched):
            if i>=32:
                break
            test_img_sample.append(img.numpy())

        test_img_sample = np.stack(test_img_sample, axis=0)
        print(f'lfw.data_sample.shape: {test_img_sample.shape}')

        self.data_splits['train'] = train_ds_unbatched
        self.data_splits['val'] = val_ds_unbatched
        self.data_splits['test'] = test_ds_unbatched
        self.data_sample = test_img_sample
        self.base_img_shape = test_img_sample.shape[1:]
    
    def get_vae_data_splits(self, batch_size):
        return [
            self.data_splits['train'].map(lambda x: (x, x)).batch(batch_size),
            self.data_splits['val'].map(lambda x: (x, x)).batch(batch_size),
            self.data_splits['test'].map(lambda x: (x, x)).batch(batch_size),
        ] 
    
    