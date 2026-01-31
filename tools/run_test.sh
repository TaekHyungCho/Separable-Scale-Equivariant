PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${your_gpu_nums, ours:2}$ --master_port=$PORT \
    $(dirname "$0")/test.py ${config_path}$ ${weight_path}$ --launcher pytorch ${@:3}