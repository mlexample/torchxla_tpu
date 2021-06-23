### Helpful Resources
* [WebDataset](https://github.com/webdataset/webdataset)
* [PyTorch/XLA](https://github.com/pytorch/xla)
* [Cloud TPU](https://cloud.google.com/tpu/docs/tpus)
* [Cloud TPU 1VM architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)

# Scaling Deep Learning Workloads with PyTorch/XLA and Cloud TPU

Overview: This end-to-end example demonstrates how to stream data from Google Cloud Storage (GCS) to a PyTorch/XLA ResNet-50 model running on a `v3-32` Cloud TPU Pod slice. In this tutorial, we use the CIFAR-10 dataset because it is publicly accesible and well known.

This tutorial uses billable components of Google Cloud, including:
* Compute Engine
* Cloud TPU
* Cloud Storage

### Setup
1. Open a Cloud Shell window.
2. Create a variable for your project's ID
```
export PROJECT_ID=REPLACE_WITH_YOURS
```
3. Configure `gcloud` command-line tool to use this project
```
gcloud config set project ${PROJECT_ID}
gcloud auth login
```
The first time you run this command in a new Cloud Shell VM, an `Authorize Cloud Shell` page is displayed. Click `Authorize` at the bottom of the page to allow `gcloud` to make GCP API calls with your credentials.

4. Create a Cloud Storage bucket. 
> **Important:** Set up your Cloud Storage bucket and TPU resources in the same region/zone to reduce network latency and network costs. This tutorial uses `europe-west4-a` 
```
export BUCKET=REPLACE_WITH_YOURS
gsutil mb -p ${PROJECT_ID} -c standard -l europe-west4 -b on gs://${BUCKET}
```
5. (optional) If you don’t use the default network, or the default network settings were edited, you may need to explicitly enable SSH access by adding a firewall rule:
```
gcloud compute firewall-rules create --network=network allow-ssh --allow=tcp:22
```
If policies in your organization/project disable these kinds of rules after a period of time, open a seperate Cloud Shell window, set your `PROJECT_ID` and run the following loop:
```
while true ; do gcloud compute firewall-rules create default-allow-ssh --allow tcp:22 ; sleep 20 ; done
```

## Create a TPU Pod slice with TPU VMs

Set the following environment variables
```
export REGION=europe-west4
export ZONE=europe-west4-a
export ACCELERATOR_TYPE=v3-32
export TPU_NAME=my-1vm-tpu
export RUNTIME_VERSION=v2-alpha
```
The following command creates the TPU Pod slice and 4 Compute Engine VMs. The metadata startup script is distributed to each VM.
```
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} \
    --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} \
    --metadata startup-script='#! /bin/bash
pip install webdataset==0.1.54
pip install google-cloud-storage
pip install tensorboardX
cd /usr/share/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
git clone --recursive https://github.com/pytorch/xla.git
git clone --recursive https://github.com/mlexample/gcspytorchimagenet.git
EOF'
```
Once the TPU VM is created, either SSH through the Cloud Console (Compute Engine > VM Instances > t1v-n-XXXXX-w-0 > SSH) or run the following command:
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```

Once in the VM, run the following command to generate the ssh-keys to ssh between VM workers on a pod:
```
gcloud compute config-ssh
```

## PyTorch Training

Check to make sure the metadata startup script has cloned all the repositories. After running the following command, we should see `gcspytorchimagenet`
```
cd /usr/share/pytorch
```
Once the repositories are visible on the VM, we are ready to train. On the VM, set the following environment variables:
```
export BUCKET=          # TODO ex: tpu-demo-xxxx
export TRAIN_SHARDS=    # TODO ex: 'train/cifar-train-{000000..000639}.tar'
export VAL_SHARDS=      # TODO ex: 'val/cifar-val-{000000..000049}.tar'

export WDS_TRAIN_DIR="pipe:gsutil cat gs://${BUCKET}/${TRAIN_SHARDS}"
export WDS_VAL_DIR="pipe:gsutil cat gs://${BUCKET}/${VAL_SHARDS}"
export LOGDIR="${LOGDIR:-gs://${BUCKET}/log-$(date '+%Y%m%d%H%M%S')}"

export TPU_NAME=my-1vm-tpu         # Name of TPU 
export NUM_EPOCHS=5                # Total number of epochs
export BATCH_SIZE=128              # Samples per train batch
export TEST_BATCH_SIZE=64          # Samples per test batch
export NUM_WORKERS=8               # Workers per TPU VM to prep/load data
export TRAIN_SIZE=1280000          # Total number of training samples
export TEST_SIZE=50000             # Total number of test samples
```
* **BUCKET**: name of GCS bucket storing our sharded dataset. We will also store training logs and model checkpoints here
* **TRAIN_SHARDS**: train shards, using brace notation to enumerate the shards
* **VAL_SHARDS**: val shards, using brace notation to enumerate the shards
* **WDS_TRAIN_DIR**: uses `pipe` to run a `gsutil` command for downloading the train shards
* **WDS_VAL_DIR**: uses `pipe` to run a `gsutil` command for downloading the val shards
* **LOGDIR**: location in GCS bucket for storing training logs

Optionally, we can pass environment variables for storing model checkpoints and loading from a previous checkpoint file:
```
export SAVE_MODEL='/tmp/model-chkpt.pt' # local file to upload to GCS
export LOAD_CHKPT_FILE=                 # object in GCS bucket 
export LOAD_CHKPT_DIR=                  # local directory/filename 
```

### Train the model
```
python3 -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME \
   --restart-tpuvm-pod-server --env XLA_USE_BF16=1 \
   -- python3 /usr/share/pytorch/REPO/test_train_mp_wds_cifar.py \
   --num_epochs=$NUM_EPOCHS \
   --batch_size=$BATCH_SIZE \
   --num_workers=$NUM_WORKERS \
   --log_steps=10 \
   --test_set_batch_size=$TEST_BATCH_SIZE \
   --wds_traindir="$WDS_TRAIN_DIR" --wds_testdir="$WDS_VAL_DIR" \
   --save_model=$SAVE_MODEL --model_bucket=$BUCKET \
   --trainsize=$TRAIN_SIZE --testsize=$TEST_SIZE \
   --logdir=$LOGDIR 2>&1 | tee -a /tmp/out-wds-1.log
```
* `--restart-tpuvm-pod-server` restarts the `XRT_SERVER` and is useful when running consecutive TPU jobs (especially if that server was left in a bad state). Since the `XRT_SERVER` is persistent for the pod setup, environment variables won’t be picked up until the server is restarted.
* `test_train_mp_wds_cifar.py` closely follows the PyTorch/XLA distributed, multiprocessing script, but is adapted to include support for WebDataset and CIFAR
* TPUs have hardware support for Brain Floating Point Format, which can be used by setting `XLA_USEBF16=1`

During training, you will see output for each logged step like this:
```
10.164.0.25 [0] | Training Device=xla:0/2 Epoch=8 Step=310 Loss=0.26758 Rate=1079.01 GlobalRate=1420.67 Time=18:02:10
```
* `10.164.0.25` refers to the IP address for this VM worker
* `[0]` refers to VM worker 0. Remember, there are 4 VM workers in our example
* `Training Device=xla:0/2` refers to the TPU core 2. In our example there are 32 TPU cores, so you should see up to `xla:0/31` (since they are 0-based)
* `Rate=1079.01` refers to the exponential moving average of examples per second for this TPU core
* `GlobalRate=1420.67` refers to the average number of examples per second for this core so far during this epoch

At the end of each epoch’s train loop, you will see output like this:
```
[0] Epoch 8 train end 18:02:10, Epoch Time=0:00:28, Replica Train Samples=39664, Reduced GlobalRate=45676.50
```
* `Replica Train Samples` tells us how many training samples this replica processed
* `Reduced GlobalRate` is the average GlobalRate across all replicas for this epoch

Once training is complete, you will see the following output log:
```
[0] Total Train Time: 0:41:25
[0] Max Accuracy: 36.84%
[0] Avg. Global Rate: 48718.11 examples per second
```

The logs for each VM worker are produced as they are available, so sometimes it can be difficult to read them sequentially. To view the logs sequentially for any TPU VM worker, we can can do the following command, where the IP_ADDRESS is the address to the left of our `[0]` 
```
grep "IP_ADDRESS" /tmp/out-wds-1.log
```
We can convert these to a `.txt` file and store them in a GCS bucket like this:
```
grep "IP_ADDRESS" /tmp/out-wds-1.log > /tmp/out-wds-1.log.txt

gsutil cp /tmp/out-wds-1.log.txt gs://${BUCKET}/YOUR_FILE_NAME.txt
```

## Cleaning up

First, disconnect from the TPU VM, if you have not already done so:
```
exit
```
In the Cloud Shell, use the following command to delete the TPU VM resources:
```
gcloud alpha compute tpus tpu-vm delete ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```
If you wish to delete the GCS bucket and its contents, run the following command in Cloud Shell:
```
gsutil rm -r gs://${BUCKET}
```
