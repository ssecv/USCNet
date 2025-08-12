class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/VOC-USC12K/'
            # return '/home/COD-SOD-dataset/SOD/DUTS'
            # return '/home/COD-SOD-dataset/COD/CAMO-TE'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
