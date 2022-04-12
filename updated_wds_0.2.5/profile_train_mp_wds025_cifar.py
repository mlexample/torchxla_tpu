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

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # cifar

cifar_img_dim = 32

def make_train_loader(cifar_img_dim, shuffle=10000,batch_size=FLAGS.batch_size):
    
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
#     wds.shuffle(1000), # TODO
    wds.decode("pil"),
    # we now have a list of decompressed train samples from each shard in this worker, in sequence
    wds.shuffle(shuffle), # TODO
    wds.to_tuple("ppm;jpg;jpeg;png", "cls"),
    wds.map_tuple(image_transform, identity),
    wds.batched(batch_size)
  ).with_epoch(epoch_size).with_length(epoch_size) # adds `__len__` method to dataset
  
  loader = wds.WebLoader(dataset, num_workers=FLAGS.num_workers, batch_size=None)
  loader = loader.with_length(epoch_size) # adds `__len__` method to dataloader
  
  return loader

# TODO
def make_val_loader(cifar_img_dim, batch_size=FLAGS.test_set_batch_size):
    
    num_dataset_instances = xm.xrt_world_size() * FLAGS.num_workers
    epoch_test_size = testsize // num_dataset_instances
    
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    val_dataset = wds.DataPipeline(
        wds.ResampledShards(FLAGS.wds_testdir),
        wds.tarfile_to_samples(),
        wds.decode("pil"),
        wds.to_tuple("ppm;jpg;jpeg;png", "cls"),
        wds.map_tuple(val_transform, identity),
        wds.batched(batch_size),
    ).with_epoch(epoch_test_size).with_length(epoch_test_size) # adds `__len__` method to dataset
    
    val_loader = wds.WebLoader(val_dataset, num_workers=FLAGS.num_workers, batch_size=None)
    val_loader = val_loader.with_length(epoch_test_size) # adds `__len__` method to dataloader
    
    return val_loader

def train_imagenet():
    print('==> Preparing data..')
    train_loader = make_train_loader(cifar_img_dim, batch_size=FLAGS.batch_size, shuffle=10000)
    test_loader = make_val_loader(cifar_img_dim, batch_size=FLAGS.test_set_batch_size)
  
    torch.manual_seed(42)
    server = xp.start_server(FLAGS.profiler_port)

    device = xm.xla_device()
    model = get_model_property('model_fn')().to(device)
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=1e-4)
    num_training_steps_per_epoch = trainsize // (
        FLAGS.batch_size * xm.xrt_world_size())
    lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
        optimizer,
        scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
        scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
        scheduler_divide_every_n_epochs=getattr(
            FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
        num_steps_per_epoch=num_training_steps_per_epoch,
        summary_writer=writer)
    loss_fn = nn.CrossEntropyLoss()
    if FLAGS.load_chkpt_file != "":
        xm.master_print("Loading saved model {}".format(FLAGS.load_chkpt_file))
        _read_blob_gcs(FLAGS.model_bucket, FLAGS.load_chkpt_file, FLAGS.load_chkpt_dir)
        checkpoint = torch.load(FLAGS.load_chkpt_dir) # torch.load(FLAGS.load_chkpt_dir)
        model.load_state_dict(checkpoint['model_state_dict']) #.to(device)
        model = model.to(device)
        
        
    def train_loop_fn(loader, epoch):
        train_steps = trainsize // (FLAGS.batch_size * xm.xrt_world_size())
        tracker = xm.RateTracker()
        total_samples = 0
        model.train()
        for step, (data, target) in enumerate(loader): 
          with xp.StepTrace('train_step', step_num=step):
            with xp.Trace('build_graph'):
              optimizer.zero_grad()
              output = model(data)
              loss = loss_fn(output, target)
              loss.backward()
              xm.optimizer_step(optimizer)
              tracker.add(FLAGS.batch_size)
              total_samples += data.size()[0]
              if lr_scheduler:
                lr_scheduler.step()
              if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                  _train_update, args=(device, step, loss, tracker, epoch, writer))
                test_utils.write_to_summary(writer, step, dict_to_write={'Rate_step': tracker.rate()}, write_xla_metrics=False)
              if step == train_steps:
                break
                
        reduced_global = xm.mesh_reduce('reduced_global', tracker.global_rate(), np.mean)
        return total_samples, reduced_global  
  
    def test_loop_fn(loader, epoch):
        test_steps = testsize // (FLAGS.test_set_batch_size * xm.xrt_world_size())
        total_samples, correct = 0, 0
        model.eval()
        for step, (data, target) in enumerate(loader):
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total_samples += data.size()[0]
            if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    test_utils.print_test_update, args=(device, None, epoch, step))
            if step == test_steps:
                break
        correct_val = correct.item()
        accuracy_replica = 100.0 * correct_val / total_samples
        accuracy = xm.mesh_reduce('test_accuracy', accuracy_replica, np.mean)
        return accuracy, accuracy_replica, total_samples
  
    # setup epoch loop
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    accuracy, max_accuracy = 0.0, 0.0
    training_start_time = time.time()
    
    if FLAGS.load_chkpt_file != "":
        best_valid_acc = checkpoint['best_valid_acc']
        start_epoch = checkpoint['epoch']
        xm.master_print('Loaded Model CheckPoint: Epoch={}/{}, Val Accuracy={:.2f}%'.format(
            start_epoch, FLAGS.num_epochs, best_valid_acc))
    else:
        best_valid_acc = 0.0
        start_epoch = 1
    
    for epoch in range(start_epoch, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(
            epoch, test_utils.now()))
        replica_epoch_start = time.time()
        
        replica_train_samples, reduced_global = train_loop_fn(train_device_loader, epoch)
        replica_epoch_time = time.time() - replica_epoch_start
        avg_epoch_time_mesh = xm.mesh_reduce('epoch_time', replica_epoch_time, np.mean)
        reduced_global = reduced_global * xm.xrt_world_size()
        xm.master_print('Epoch {} train end {}, Epoch Time={}, Replica Train Samples={}, Reduced GlobalRate={:.2f}'.format(
            epoch, test_utils.now(), 
            str(datetime.timedelta(seconds=avg_epoch_time_mesh)).split('.')[0], 
            replica_train_samples, 
            reduced_global))

        accuracy, accuracy_replica, replica_test_samples = test_loop_fn(test_device_loader, epoch)
        xm.master_print('Epoch {} test end {}, Reduced Accuracy={:.2f}%, Replica Accuracy={:.2f}%, Replica Test Samples={}'.format(
            epoch, test_utils.now(), 
            accuracy, accuracy_replica, 
            replica_test_samples))
        
        if FLAGS.save_model != "":
            if accuracy > best_valid_acc:
                xm.master_print('Epoch {} validation accuracy improved from {:.2f}% to {:.2f}% - saving model...'.format(epoch, best_valid_acc, accuracy))
                best_valid_acc = accuracy
                xm.save(
                    {
                        "epoch": epoch,
                        "nepochs": FLAGS.num_epochs,
                        "model_state_dict": model.state_dict(),
                        "best_valid_acc": best_valid_acc,
                        "opt_state_dict": optimizer.state_dict(),
                    },
                    FLAGS.save_model,
                )
                if xm.is_master_ordinal():
                    _upload_blob_gcs(FLAGS.logdir, FLAGS.save_model, 'model-chkpt.pt')
                            
        max_accuracy = max(accuracy, max_accuracy)
        test_utils.write_to_summary(
            writer,
            epoch,
            dict_to_write={'Accuracy/test': accuracy,
                           'Global Rate': reduced_global},
            write_xla_metrics=False)
        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())
    test_utils.close_summary_writer(writer)
    total_train_time = time.time() - training_start_time
    xm.master_print('Total Train Time: {}'.format(str(datetime.timedelta(seconds=total_train_time)).split('.')[0]))    
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    xm.master_print('Avg. Global Rate: {:.2f} examples per second'.format(reduced_global))
    return max_accuracy


def _mp_fn(index, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    accuracy = train_imagenet()
    if accuracy < FLAGS.target_accuracy:
        print('Accuracy {} is below target {}'.format(accuracy,
                                                      FLAGS.target_accuracy))
        sys.exit(21)


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores, start_method='fork') # , start_method='spawn'
