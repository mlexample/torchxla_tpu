### WebDataset==0.2.5

To adapt the original script to this version of the [WebDataset API](https://github.com/webdataset/webdataset), we make changes to the `make_train_loader` and `make_val_loader` functions:

```
def make_train_loader(cifar_img_dim, shuffle=10000,batch_size=FLAGS.batch_size):
    
  num_dataset_instances = xm.xrt_world_size() * FLAGS.num_workers
  epoch_size = trainsize // num_dataset_instances
  
  image_transform = transforms.Compose(
    [
      transforms.RandomCrop(cifar_img_dim),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ]
  )
  
  dataset = wds.DataPipeline(
    wds.ResampledShards(FLAGS.wds_traindir),
    # we now have an iterator over all shards
    wds.tarfile_to_samples(),
    wds.decode("pil"),
    # we now have a list of decompressed train samples from each shard in this worker, in sequence
    wds.to_tuple("ppm;jpg;jpeg;png", "cls"),
    wds.map_tuple(image_transform, identity),
    wds.batched(batch_size)
  ).with_epoch(epoch_size).with_length(epoch_size) # adds `__len__` method to dataset
  
  loader = wds.WebLoader(dataset, num_workers=FLAGS.num_workers, batch_size=None)
  loader = loader.with_length(epoch_size) # adds `__len__` method to dataloader
  
  return loader
```

* `with_epoch` sets the epoch length explicitly 
* As the Webdataset is a custom `IterableDataset` we explictly add a `__len__` method to the dataset and dataloader using the `with_length` method ([source](https://github.com/webdataset/webdataset/blob/05a1ea1116781ffe3c3bc257061f2f3e51dfeb0b/webdataset/pipeline.py#L96))


### Specify WebDataset API version in the TPU-VM metadata startup script
```
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} \
    --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} \
    --metadata startup-script='#! /bin/bash
pip install webdataset==0.2.5
pip install google-cloud-storage
pip install tensorboardX
cd /usr/share/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
git clone --recursive https://github.com/pytorch/xla.git
git clone --recursive https://github.com/mlexample/torchxla_tpu.git
EOF'
```
