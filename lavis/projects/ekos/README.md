

### Pretraining

#### Training
```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/epic-kitchens/pretrain.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/pretrain.yaml
```
* ekos Checkpoint: `EK_PT_CHECKPOINT = lavis/output/results/epic_kitchens/epic_kitchens_pretrain/{job_id}/checkpoint_best.pth`
* s20bn Checkpoint: `SS_PT_CHECKPOINT = lavis/output/results/s20bn/s20bn_pretrain/{job_id}/checkpoint_best.pth`

#### In-Domain Evaluation
```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/epic-kitchens/eval_pretrain/eval-ekos.yaml \
       --options run.resume_ckpt_path=$EK_PT_CHECKPOINT
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/eval_pretrain/eval-s20bn.yaml \
       --options run.resume_ckpt_path=$SS_PT_CHECKPOINT
```

#### Cross-Evaluation
```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/epic-kitchens/eval_pretrain/eval-s20bn.yaml \
       --options run.resume_ckpt_path=$EK_PT_CHECKPOINT
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/eval_pretrain/eval-ekos.yaml \
       --options run.resume_ckpt_path=$SS_PT_CHECKPOINT
```

### Finetuning

#### Training
```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/epic-kitchens/vqa_finetune.yaml \
       --options run.resume_ckpt_path=$EK_PT_CHECKPOINT
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/vqa_finetune.yaml \
       --options run.resume_ckpt_path=$SS_PT_CHECKPOINT
```
* ekos Checkpoint: `EK_FT_CHECKPOINT = lavis/output/results/epic_kitchens/epic_kitchens_vqa_finetune/{job_id}/checkpoint_best.pth`
* s20bn Checkpoint: `SS_FT_CHECKPOINT = lavis/output/results/s20bn/s20bn_vqa_finetune/{job_id}/checkpoint_best.pth`

* Results: `lavis/output/results/epic_kitchens/epic_kitchens_vqa_finetune/{job_id}/result/val_epic_kitchens_result_{epoch}.json`
* Results: `lavis/output/results/epic_kitchens/s20bn_vqa_finetune/{job_id}/result/val_epic_kitchens_result_{epoch}.json`

#### In-Domain Evaluation
```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/epic-kitchens/eval_finetune/eval-ekos.yaml \
       --options run.resume_ckpt_path=$EK_FT_CHECKPOINT
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/eval_finetune/eval-s20bn.yaml \
       --options run.resume_ckpt_path=$SS_FT_CHECKPOINT
```

#### Cross-Evaluation
```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/epic-kitchens/eval_finetune/eval-s20bn.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/eval_finetune/eval-ekos.yaml
```

### Overfitting
```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/overfit/pretrain-overfit.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/overfit/pretrain-overfit-eval.yaml
# get 
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/overfit/overfit.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 evaluate.py --cfg-path lavis/projects/ekos/s20bn/overfit/overfit-eval.yaml
```