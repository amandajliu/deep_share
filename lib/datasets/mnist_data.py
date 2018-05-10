# import base class
from imdb import Imdb
import numpy as np
import os 
import cPickle
from utils.error import compute_mle, compute_neg_mle

"""Class to manipulate DeepFashion dataset """

class MnistData(Imdb):
    """ Image database for DeepFashion dataset. """
    
    def __init__(self, split):
        name = 'mnist_'+split      
        Imdb.__init__(self, name)
                
        # load image paths
        self._data_path = os.path.join(self.data_path, 'imdb_mnist') 
        
        # attribute classes
        self._classes = [0,1,2,3,4,5,6,7,8,9]

        # load annotations
        self._load_dataset(split)

    def _load_dataset(self, split):
        """ Load image path list and ground truths """
        
        # load image path list and ground truths from the cache
        cache_file = os.path.join(self.cache_path, self.name+'_dbcache.pkl')
        
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         dbcache = cPickle.load(fid)
        #         print '{} database cache loaded from {}'.format(self.name, cache_file)
        #         self._image_list = dbcache['image_list']
        #         self._gtdb = dbcache['gtdb']
        #         return
        
        # load list of images
        self._image_list, self._index_list = self._do_load_filelist(split)   
        
        # load attributes and landmarks data
        self._gtdb = {'attr': np.zeros((self.num_images, self.num_classes), dtype=np.bool)}    
        
      
        file = os.path.join(self.data_path, '{}ing_img_labels.txt'.format(split))
        with open(file) as f:
            for line in f:
                info = line.split()
                img_idx = int(info[0].split('.')[0])
                label = int(info[1])
                self._gtdb['attr'][img_idx, label] = 1



        dbcache = {'image_list': self.image_list, 'gtdb': self.gtdb}
        # save to cache         
        with open(cache_file, 'wb') as fid:
            cPickle.dump(dbcache, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote database cache to {}'.format(cache_file)

    def _do_load_filelist(self, split):
        """ Return the absolute paths to image files """
        # file = os.path.join(self.data_path, 'Eval', 'list_eval_partition.txt')
        # # list of keywords associated with the desired partition
        # if split == 'train':
        #     sp_idx = ['train']
        # # elif split == 'val':
        # #     sp_idx = ['val']
        # elif split == 'test':
        #     sp_idx = ['test']
        # # elif split == 'trainval':
        # #     sp_idx = ['train', 'val']
        # else:
        #     raise NameError('Undefined Data Split: {}'.format(split))           

        basepath = self.data_path
        file = os.path.join(self.data_path, '{}ing_img_labels.txt'.format(split))
        # find paths to all files
        image_list = []
        idx_list = []
        idx = 0

        with open(file) as f:
            for line in f:
                info = line.split()
                image_list.append(os.path.join(self.data_path, '{}ing'.format(split), info[0]))
                idx_list.append(idx)
                idx += 1

        # if split == 'train':
        #     for train_image in range(len(mnist.train_images())):
        #         idx_list.append(idx)
        #         image_list.append("train_"+str(idx))
        #         idx += 1
        # elif split == 'test':
        #     for test_image in range(len(mnist.test_images())):
        #         idx_list.append(idx)
        #         image_list.append("test_"+str(idx))
        #         idx += 1
        # else:
        #     raise NameError('Undefined Data Split: {}'.format(split)) 

        # with open(file, 'r') as fid:
        #     # skip first two lines
        #     next(fid)
        #     next(fid)
        #     # parse the ground truth file
        #     for line in fid:
        #         split = line.split()
        #         if split[1] in sp_idx:
        #             image_list.append(os.path.join(basepath, split[0]))
        #             idx_list.append(idx)
        #         idx = idx + 1
        
        return image_list, idx_list

    # def _do_load_attributes(self):
    #     """ Load attributes of the listed images. """
    #     file = os.path.join(self.data_path, 'Anno', 'list_category_img.txt')

    #     attr = np.zeros((self.num_images, self.num_classes), dtype=np.bool)


    #     with open(file, 'r') as fid:
    #         # skip the first two lines
    #         next(fid)
    #         next(fid)
    #         # parse the ground truth file
    #         for line in fid:
    #             split = line.split()
    #             name_i = os.path.join(self.data_path, split[0])
    #             idx = [i for i in xrange(self.num_images) if self._image_list[i]==name_i]
    #             if len(idx) > 0:
    #                 cls_id = int(split[1]) - 1
    #                 attr[idx[0], cls_id] = 1.0
    #                 # attr[idx[0], :] = np.array(split[1:], dtype=np.float32) > 0
    #     return attr
    
    def evaluate(self, scores, ind, cls_idx=None, eval_type=None):
        """ Evaluation: Report classification error rate. 
            "scores" is a (N x C) matrix, where N is the number of samples, 
                C is the number of classes. C=len(cls_idx) if provided.
            "ind" is an array that index into result
        """
        if cls_idx is None:
            cls_idx = np.arange(self.num_classes)

        gt = self.gtdb['attr'][ind, :]
        gt = gt[:, cls_idx]
	if eval_type == "neg":
            err = compute_neg_mle(scores, gt, False)
        elif eval_type == "pos":
            err = compute_neg_mle(scores, gt, True)
        else:
            err = compute_mle(scores, gt)
        
        return err

    def print_info(self, i):
        """ Output information about the image and some ground truth. """

        im_size = self.image_size(i)
        print 'The path of the image is: {}'.format(self.image_path_at(i))
        print 'width: {}, height: {}'.format(im_size[0], im_size[1])
        
        attr_i = self.gtdb['attr'][i, :]
        print 'The attributes are: {}'.format(','.join([self._classes[i] for i in np.where(attr_i==1)[0]]))
