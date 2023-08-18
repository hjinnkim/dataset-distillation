#!/bin/sh
#SBATCH -J DD_v2_4 ## Job Name
#SBATCH -p gpu_3090            ## Partition name
#SBATCH -N 1 ## node count 총필요한컴퓨팅노드수
#SBATCH -n 4 ## total number of tasks across all nodes 총필요한프로세스수
#SBATCH -o /home/p109g2208/hyeonjin/dataset-distillation/slurm/logs/o/%x.o%j ## filename of stdout, stdout 파일 명(.o)
#SBATCH -e /home/p109g2208/hyeonjin/dataset-distillation/slurm/logs/e/%x.e%j ## filename of stderr, stderr 파일 명(.e)
#SBATCH --time 1:00:00  ## 최대 작업 시간(Wall Time Clock Limit)
#SBATCH --gres=gpu:1 ## number of GPU(s) per node


source activate torch38
python main.py --mode distill_basic --dataset Cifar10_simCLR --arch ResNet18SimCLR --distilled_images_per_class_per_step 8 --results_dir ./results_v2_4/