import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class LfwDataset:
    def __init__(self):
        self.data_splits = {}
        self.data_sample = None
    
    @staticmethod
    def scale_and_crop_img(ds_iter):
        return (tf.cast(ds_iter['image'][45:205, 45:205, :], dtype='float32') / 255.0)

    @staticmethod
    def unscale_img(img):
        return tf.squeeze(img)
    
    def tfds_load(self):
        train_ds_unbatched, train_info = tfds.load('lfw', split='train[:80%]', with_info=True)
        val_ds_unbatched, val_info = tfds.load('lfw', split='train[80%:90%]', with_info=True)
        test_ds_unbatched, test_info = tfds.load('lfw', split='train[90%:]', with_info=True)
        sizes = []


        train_ds_unbatched = train_ds_unbatched.map(LfwDataset.scale_and_crop_img).shuffle(1000)
        val_ds_unbatched = val_ds_unbatched.map(LfwDataset.scale_and_crop_img)
        test_ds_unbatched = test_ds_unbatched.map(LfwDataset.scale_and_crop_img)

        samples = []
        for ds in [train_ds_unbatched, val_ds_unbatched, test_ds_unbatched]:
            print(ds)
            for i,batch in enumerate(ds):
                if i < 3:
                    samples.append(batch)
                    print(f'{i:04d}', batch.shape)
                pass
            sizes.append(i+1)

        print('Train samples:', sizes[0], type(train_ds_unbatched))
        print('Val samples:  ', sizes[1], type(val_ds_unbatched))
        print('Test samples: ', sizes[2], type(test_ds_unbatched))

        test_img_sample = []
        for i,img in enumerate(test_ds_unbatched):
            if i>=32:
                break
            test_img_sample.append(img.numpy())

        test_img_sample = np.stack(test_img_sample, axis=0)
        print(f'Display test image shape: {test_img_sample.shape}')

        self.data_splits['train'] = train_ds_unbatched
        self.data_splits['val'] = val_ds_unbatched
        self.data_splits['test'] = test_ds_unbatched
        self.data_sample = test_img_sample
    
    
    