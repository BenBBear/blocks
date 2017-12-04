#!/usr/bin/env bash

declare -a tasks4=()
declare -a tasks2=()
declare -a tasks1=("ssd/train_voc-resnet18-1bit-default" "ssd/train_voc-resnet18-2bit-default")

for name in  "${tasks4[@]}"
do
echo "***********************  submitting $name  ***********************"
bash ./examples/scripts/submit.sh $name examples 4 &
done

for name in  "${tasks2[@]}"
do
echo "***********************  submitting $name  ***********************"
bash ./examples/scripts/submit.sh $name examples 2 &
done

for name in  "${tasks1[@]}"
do
echo "***********************  submitting $name  ***********************"
bash ./examples/scripts/submit.sh $name examples 1 &
done

echo "Waiting all submission to finish"
wait
echo "Done"