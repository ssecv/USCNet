class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/desktop/VOC-USC12K/'
            # return '/home/desktop/COD-SOD-dataset/SOD/PASCAL-S'
            # return '/home/desktop/COD-SOD-dataset/COD/COD10K'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
