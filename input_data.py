import numpy
import os
def dense_to_one_hot(labels_dense, num_classes=10):

  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      images = images.astype(numpy.float32)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data(data_file_name, one_hot, skiprows = 1, label_clo = 1, img_clo_st = 3):
    data = numpy.loadtxt(data_file_name, dtype=numpy.str, delimiter=",", skiprows=skiprows)
    imgs = data[:, img_clo_st:].astype(numpy.float32)
    labels = data[:, label_clo].astype(numpy.int8)
    if one_hot:
        labels = dense_to_one_hot(labels, 2)
    return imgs, labels

def read_data_sets(train_dir, setFileNames = ['740-Data1.csv', 'test-2.csv', '1953.csv'], fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets
  TRAIN_FILE = os.path.join(train_dir, setFileNames[0])
  VALIDATION_FILE = os.path.join(train_dir, setFileNames[1])
  TEST_FILE = os.path.join(train_dir, setFileNames[2])
  train_images, train_labels = read_data(data_file_name=TRAIN_FILE, skiprows=1, label_clo = 1, img_clo_st = 2, one_hot=one_hot)
  validation_images, validation_labels = read_data(data_file_name=VALIDATION_FILE, skiprows=1, label_clo = 1, img_clo_st = 2, one_hot=one_hot)
  test_images, test_labels = read_data(data_file_name=TEST_FILE, skiprows=1, label_clo = 1, img_clo_st = 2, one_hot=one_hot)

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets
