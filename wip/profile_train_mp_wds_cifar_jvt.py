import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.metrics as met
import torch_xla
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sys
import os
import webdataset as wds
import datetime
import time
from itertools import islice
import torch_xla.debug.profiler as xp
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.blob import Blob
# import torch_xla.utils.serialization as xser


for extra in ('/usr/share/torch-xla-1.8/pytorch/xla/test', '/pytorch/xla/test', '/usr/share/pytorch/xla/test'):
    if os.path.exists(extra):
        sys.path.insert(0, extra)

import schedulers
import args_parse 

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
    '--dataset': {
        'choices': ['gcsdataset', 'torchdataset'],
        'default': 'gcsdataset',
        'type': str,
    },
    '--wds_traindir': {
        'type': str,
        'default':'/tmp/cifar',
    },
    '--wds_testdir': {
        'type': str,
        'default': '/tmp/cifar',
    },
    '--trainsize': {
        'type': int,
        'default': 1280000,
    },
    '--testsize': {
        'type': int,
        'default': 50000,
    },
    '--save_model': {
        'type': str,
        'default': "",
    },
    '--load_chkpt_file': {
        'type': str,
        'default': "",
    },
    '--load_chkpt_dir': {
        'type': str,
        'default': "",
    },
    '--model_bucket': {
        'type': str,
        'default': "",
    },
}
        
FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    opts=MODEL_OPTS.items(),
    profiler_port=9012,
)


DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            DEFAULT_KWARGS, **{
                'lr': 0.5,
                'lr_scheduler_divide_every_n_epochs': 20,
                'lr_scheduler_divisor': 5,
                'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
            })
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
    if getattr(FLAGS, arg) is None:
        setattr(FLAGS, arg, value)

def get_model_property(key):
    default_model_property = {
        'img_dim': 224,
        'model_fn': getattr(torchvision.models, FLAGS.model)
    }
    model_properties = {
        'inception_v3': {
            'img_dim': 299,
            'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
        },
    }
    model_fn = model_properties.get(FLAGS.model, default_model_property)[key]
    return model_fn


def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)

trainsize = FLAGS.trainsize 
testsize = FLAGS.testsize 

def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
    """Uploads a file to GCS bucket"""
    client = storage.Client()
    blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
    blob.bucket._client = client
    blob.upload_from_filename(source_file_name)
    
    xm.master_print("Saved Model Checkpoint file {} and uploaded to {}.".format(source_file_name, os.path.join(gcs_uri, destination_blob_name)))
    
def _read_blob_gcs(BUCKET, CHKPT_FILE, DESTINATION):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET)
    blob = bucket.get_blob(CHKPT_FILE)
    blob.download_to_filename(DESTINATION)
    
def identity(x):
    return x
  
# my_worker_splitter

# my_node_splitter

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # cifar

cifar_img_dim = 32

def make_train_loader(cifar_img_dim, shuffle=10000, batch_size=FLAGS.batch_size):
  num_dataset_instances = xm.xrt_world_size() * FLAGS.num_workers
  epoch_size = trainsize // num_dataset_instances
  
  image_transform = transforms.Compose(
    [
      transforms.RandomCrop(cifar_img_dim), # padding=4
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ]
  )
  
  dataset = wds.DataPipeline(
    wds.ResampledShards(FLAGS.wds_traindir),
    # we now have an iterator over all shards
    wds.tarfile_to_samples(),
    wds.shuffle(1000),
    wds.decode("pil"),
    # we now have a list of decompressed train samples from each shard in this worker, in sequence
    wds.shuffle(10000),
    wds.to_tuple("ppm;jpg;jpeg;png", "cls"),
    wds.map_tuple(image_transform, identity),
    wds.batched(batch_size)
  ).with_epoch(num_dataset_instances)
  
  loader = wds.WebLoader(dataset, num_workers=FLAGS.num_workers, batch_size=None)
  
  return loader

# TODO
    

  
  
  
  
  
  
  
