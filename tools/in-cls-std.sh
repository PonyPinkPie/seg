CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/cls-std/seed-0.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/cls-std/seed-1024.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/cls-std/seed-2048.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/cls-std/seed-4096.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/cls-std/seed-8192.json

CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/in-std/seed-0.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/in-std/seed-1024.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/in-std/seed-2048.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/in-std/seed-4096.json
CUDA_VISIBLE_DEVICES=1 python tools/train.py --cfg config/in-std/seed-8192.json

