#!/usr/bin/env bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCAL_HADOOP_HOME/lib/native:$LOCAL_JAVA_HOME/lib/amd64/server:/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export HDFS=hdfs://hobot-bigdata/

JOB_NAME=$1
echo "JOB_NAME IS ${JOB_NAME}"

host=`hostname`
ids=`date +%s`
job_id="${JOB_NAME}${ids}"
echo "Starting Job Submit. Jobid == ${job_id}....."

HADOOP_OUT=/open_mlp/run_data/output/${job_id}

qsub_i --conf /etc/qsub_i.conf \
    -N ${job_id} \
    --hdfs $HDFS --ugi xinyuzhang,xinyuzhang \
    --hout $HADOOP_OUT \
    --files $2 \
    --pods ${3} \
    -l walltime=240:00:00 \
    ./$2/job.sh

echo "Your job_name : ${job_id}, submitted!"