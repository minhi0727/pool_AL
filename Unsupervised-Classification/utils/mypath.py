"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'mnist', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200', 'tinyimagenet', 'imbalanced-cifar-10', 'imbalanced-cifar-100'}
        assert(database in db_names)

        if database in ['cifar-10', 'imbalanced-cifar-10']:
            return '/home/chominhi/work/init-pools-dal-main/repository_eccv/cifar-10/'
        
        elif database == 'cifar-20':
            return '/home/chominhi/work/init-pools-dal-main/repository_eccv/cifar-20/'

        elif database == 'stl-10':
            return '/home/chominhi/work/init-pools-dal-main/repository_eccv/stl-10/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '/path/to/imagenet/'
        
        elif database == 'mnist':
            return '../data/mnist/'
        
        elif database == 'tinyimagenet':
            return '/DATA/akshay/datasets/tiny-imagenet-200/'
        
        else:
            raise NotImplementedError
