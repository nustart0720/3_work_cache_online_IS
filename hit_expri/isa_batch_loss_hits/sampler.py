import torch
from torch._six import int_classes as _int_classes
import math,random,time,operator

class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples

class Online_IS_Sampler(Sampler):
    r"""Use ICLR 2016 online batch selection, default argument is set like this:
    Ts = batch_size; r(ratio) = 1.0; s(0) = 10^2, s(end) = 10^2; recompute_loss frequency = 0.5

    """
    global_epoch = 0 # class var


    def __init__(self, data_source, replacement=True, s_0 = 1E2, s_end=1E2, recompute_loss_freq = 0.5):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.replacement = replacement

        self.se_decline_factor = math.exp(math.log(s_end/s_0)/(s_end-s_0)) if s_end!=s_0 else 1
        print('self.se_decline_factor',self.se_decline_factor)
        self.recompute_loss_freq = recompute_loss_freq

        self.latest_loss = {idx:math.inf for idx in range(len(data_source))} # k:v -> imgidx:latest_loss
        self.sorted_lst = [] # 对self.latest_loss按值排序后的列表，其中内容为元组，第一个元素是idx，第二个是loss
        self.p_list = [1]*self.num_samples
        self.accu_lst = [0]*self.num_samples

        self.s_0 = s_0

    def __iter__(self):
        st = time.time()
        if self.global_epoch == 0:
            self.global_epoch += 1
            return iter(torch.randperm(self.num_samples).tolist())
        else:
            se = math.pow(self.se_decline_factor, self.global_epoch) # 最大和最小概率之间的倍数关系是s_0*se
            # print('se',se)
            # print('get se:',time.time()-st)
            # 对latest_loss_lst按值排序，从大到小
            # self.sorted_lst = sorted(self.latest_loss.items(), key = lambda kv:(kv[1],kv[0])) # 32 seconds
            self.sorted_lst = [(k,v) for k,v in sorted(self.latest_loss.items(), key=operator.itemgetter(1), reverse=True)] # 16 seconds -- 因为没有先从cuda移动出来
            # print('self.sorted_lst:',self.sorted_lst[0],self.sorted_lst[-1])

            print('sort self.latest_loss done',time.time()-st)
            # 采用概率选择，产生一个epoch长度的候训练数据idx
            epoch_idx = self.get_epoch_idx_by_prob(self.s_0*se)
            print('get epoch idx by prob',time.time()-st)
            # print(epoch_idx[:10])
            self.global_epoch += 1
            print('sampler index set len:',len(set(epoch_idx)))
            return iter(epoch_idx)

    
    def update_loss(self, dic_imgidx_loss):
        for imgidx, loss in dic_imgidx_loss.items():
            self.latest_loss[imgidx] = loss
    
    def get_epoch_idx_by_prob(self,se):
        mult = math.exp(math.log(se)/self.num_samples) # 各pi之间的倍数
        print('mul=',mult)
        # 填充self.p_list
        for i in range(self.num_samples):
            if i == 0:
                self.p_list[i] = 1
            else:
                self.p_list[i] = self.p_list[i-1] / mult
        # 正则化
        p_sum = sum(self.p_list)
        self.p_list = [pi/p_sum for pi in self.p_list]
        # 填写self.accu_lst
        for i in range(self.num_samples):
            if i == 0 :
                self.accu_lst[i] = self.p_list[i]
            else:
                self.accu_lst[i] = self.accu_lst[i-1] + self.p_list[i]
        # print('accu_lst calculated')
        selected_epoch_idx = []
        
        # 重复self.num_samples次，每次随机产生一个数r介于[0,1)，找出最小的idx使得a[idx]>=r
        for _ in range(int(self.num_samples*0.5)):
            r = random.random()
            idx = self.find_minidx_from_accu(r)
            # 根据accu_lst从self.sorted_lst中选出目的图片的imgidx
            selected_epoch_idx.append(self.sorted_lst[idx][0])
        '''
        for _ in range(256*8):
            r = random.random()
            idx = self.find_minidx_from_accu(r)
            # 根据accu_lst从self.sorted_lst中选出目的图片的imgidx
            selected_epoch_idx.append(self.sorted_lst[idx][0])
        '''
        # print('selected_epoch_idx done')
        random.shuffle(selected_epoch_idx)
        return selected_epoch_idx

    def find_minidx_from_accu(self, target):
        left = 0
        right = len(self.accu_lst)
        while left <= right:
            mid = (right - left) // 2 + left
            if self.accu_lst[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return mid
    
    def __len__(self):
        return self.num_samples


class Online_IS_Sampler_version2(Sampler):
    r"""Use ICLR 2016 online batch selection, default argument is set like this:
    Ts = batch_size; r(ratio) = 1.0; s(0) = 10^2, s(end) = 10^2; recompute_loss frequency = 0.5

    """
    global_epoch = 0 # class var


    def __init__(self, data_source, replacement=True, s_0 = 1E2, s_end=1E2, recompute_loss_freq = 0.5):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.replacement = replacement

        self.se_decline_factor = math.exp(math.log(s_end/s_0)/(s_end-s_0)) if s_end!=s_0 else 1
        self.recompute_loss_freq = recompute_loss_freq

        self.latest_loss = [math.inf] * len(data_source) # idx是imgidx 值是latest_loss
        self.sorted_idx = [] # 对self.latest_loss按值排序后的列表
        self.p_list = [1]*self.num_samples
        self.accu_lst = [0]*self.num_samples

    def __iter__(self):
        st = time.time()
        if self.global_epoch == 0:
            self.global_epoch += 1
            return iter(torch.randperm(self.num_samples).tolist())
        else:
            se = math.pow(self.se_decline_factor, self.global_epoch) # 最大和最小概率之间的倍数关系
            print('get se:',time.time()-st)
            # 对latest_loss_lst按值排序，从大到小
            # self.sorted_lst = sorted(self.latest_loss.items(), key = lambda kv:(kv[1],kv[0])) # 32 seconds
            # self.sorted_lst = [(k,v) for k,v in sorted(self.latest_loss.items(), key=operator.itemgetter(1))] # 16 seconds
            self.sorted_idx = sorted(range(self.num_samples), key = lambda k:self.latest_loss[k], reverse=True)

            print('sort self.latest_loss done',time.time()-st)
            # 采用概率选择，产生一个epoch长度的候训练数据idx
            epoch_idx = self.get_epoch_idx_by_prob(se)
            print('get epoch idx by prob',time.time()-st)
            print(epoch_idx[:10])
            self.global_epoch += 1

            return iter(epoch_idx)

    
    def update_loss(self, dic_imgidx_loss):
        for imgidx, loss in dic_imgidx_loss.items():
            self.latest_loss[imgidx] = loss
    
    def get_epoch_idx_by_prob(self,se):
        mult = math.exp(math.log(se)/self.num_samples) # 各pi之间的倍数
        # 填充self.p_list
        for i in range(self.num_samples):
            if i == 0:
                self.p_list[i] = 1
            else:
                self.p_list[i] = self.p_list[i-1] / mult
        # 正则化
        p_sum = sum(self.p_list)
        self.p_list = [pi/p_sum for pi in self.p_list]
        # 填写self.accu_lst
        for i in range(self.num_samples):
            if i == 0 :
                self.accu_lst[i] = self.p_list[i]
            else:
                self.accu_lst[i] = self.accu_lst[i-1] + self.p_list[i]
        print('accu_lst calculated')
        selected_epoch_idx = []
        
        # 重复self.num_samples次，每次随机产生一个数r介于[0,1)，找出最小的idx使得a[idx]>=r
        for _ in range(self.num_samples):
            r = random.random()
            idx = self.find_minidx_from_accu(r)
            # 根据accu_lst从self.sorted_idx中选出目的图片的imgidx
            selected_epoch_idx.append(self.sorted_idx[idx])

        print('selected_epoch_idx done')
        random.shuffle(selected_epoch_idx)
        return selected_epoch_idx

    def find_minidx_from_accu(self, target):
        left = 0
        right = len(self.accu_lst)
        while left <= right:
            mid = (right - left) // 2 + left
            if self.accu_lst[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return mid
    
    def __len__(self):
        return self.num_samples



class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
