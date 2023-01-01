module load CUDA/9.2.88-GCC-7.3.0-2.30
srun --ntasks=1 --cpus-per-task=4 -t 1:30:00 --gres=gpu:1 -A p_lv_hpgpu2223 --pty bash -i