import os
import shutil
import re
from tqdm import tqdm
from functools import partial

from omegaconf import OmegaConf, DictConfig

# PyTorch
import torch
from torch.nn import CrossEntropyLoss

# Hugging Face
from datasets import load_from_disk, load_dataset
from transformers import BatchEncoding
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs
from datasets import Features, Sequence, Value, IterableDataset

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param

# Our metric object and model
from .metrics import Metrics
from ..model import NeoBERTLMHead, NeoBERTConfig
from ..tokenizer import get_tokenizer
from ..tokenizer.tokenizer import single_column_mapping
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..dataloader import get_dataloader
from .entropy_regularization import compute_head_entropy

def to_target_batch_size(
    batch: BatchEncoding,
    stored_batch: BatchEncoding,
    target_size: int = 8,
):
    tmp = {}
    batch_size = batch["input_ids"].shape[0]

    # If the batch is to large, we store samples
    if batch_size > target_size:
        for key in batch.keys():
            tmp[key] = torch.split(batch[key], [target_size, batch_size - target_size], dim=0)
            batch[key] = tmp[key][0]
            stored_batch[key] = tmp[key][1] if stored_batch[key] is None else torch.cat([tmp[key][1], stored_batch[key]], dim=0)

    # If the batch is to small, we fetch stored samples
    elif batch_size < target_size and stored_batch["input_ids"] is not None:
        stored_batch_size = stored_batch["input_ids"].shape[0]
        missing = target_size - batch_size

        # Fetch only necessary samples if storage is larger than required
        if missing < stored_batch_size:
            for key in batch.keys():
                stored_batch[key].to(batch[key].device)
                tmp[key] = torch.split(stored_batch[key], [missing, stored_batch_size - missing], dim=0)
                batch[key] = torch.cat([batch[key], tmp[key][0]], dim=0)
                stored_batch[key] = tmp[key][1]
                stored_batch[key].to("cpu", non_blocking=True)

        # Concatenate otherwise
        else:
            for key in batch.keys():
                batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                stored_batch[key] = None

    return batch, stored_batch


def trainer(cfg: DictConfig):
    # Get the last checkpoint id
    checkpoint_dir = os.path.join(cfg.trainer.dir, "checkpoints")
    dataset_checkpoint_dir = os.path.join(cfg.trainer.dir, "dataset_checkpoint")
    model_checkpoint_dir = os.path.join(cfg.trainer.dir, "model_checkpoints")
    os.makedirs(dataset_checkpoint_dir, exist_ok=True)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    iteration = 0
    if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        # This regular expression was taken from accelerator.load_state()
        folders = os.listdir(checkpoint_dir)
        iteration = max(int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in folders) + 1

    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.accelerate.max_ckpt,
        iteration=iteration,
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": OmegaConf.to_container(cfg) | {"distributed_type": accelerator.distributed_type},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "resume": cfg.wandb.resume,
            }
        },
    )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg.trainer.tf32

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16
    
    print("loading tokenizer")
    # Tokenizer
    tokenizer = get_tokenizer(**cfg.tokenizer)
    
    print("loading dataset")
    # Dataset
    if cfg.dataset.name == "refinedweb" :
        dataset = load_dataset(cfg.dataset.train.path, streaming=True, split="train")

        features = Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("bool"))})
        mapping = partial(single_column_mapping, tokenizer=tokenizer, column_name=cfg.dataset.column, max_length=cfg.tokenizer.max_length, truncation=True)
        train_dataset= dataset.shuffle(seed=42, buffer_size=10_000)
        train_dataset = train_dataset.map(
            mapping,
            batched=True,
            remove_columns=dataset.column_names,
            features=features,
        )

        # if cfg.trainer.resume and os.path.exists(dataset_checkpoint_dir) and len(os.listdir(dataset_checkpoint_dir)) > 0:
        #     print("starting dataset from state dict")
        #     dataset_state_dict = torch.load(os.path.join(dataset_checkpoint_dir, "iterableDataset.pt"))
        #     print(dataset_state_dict)
        #     dataset.load_state_dict(dataset_state_dict)
    else :
        train_dataset = load_from_disk(cfg.dataset.path_to_disk)
    
    # Dataloader
    train_dataloader = get_dataloader(train_dataset, tokenizer, dtype=dtype_pad_mask, **cfg.dataloader.train, **cfg.datacollator)
    
    print("loading model")
    # Model
    model = NeoBERTLMHead(NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id))
    accelerator.log({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    
    print("optim and schedul")
    # Optimizer and Scheduler
    optimizer = get_optimizer(model, accelerator.distributed_type, name=cfg.optimizer.name, **cfg.optimizer.hparams)
    scheduler = get_scheduler(optimizer=optimizer, lr=cfg.optimizer.hparams.lr, **cfg.scheduler)
    
    print("preparing accelerator")
    # Prepare with accelerate
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,
        model,
        optimizer,
        scheduler,
    )

    # Loss function
    train_loss_fn = CrossEntropyLoss()

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        print("skipping examples")
        accelerator.load_state()
        train_dataloader.set_epoch(metrics["train/epochs"])
        skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["train/batches"] % (cfg.trainer.max_steps//cfg.dataloader.train.batch_size))
        print("done")


    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["train/steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )
    
    print("launching")

    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = train_dataloader if skipped_train_dataloader is None else skipped_train_dataloader

        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        i = 0
        for id, batch in enumerate(dataloader):
            # print("batch", id)
            # Update number of batches
            metrics["train/batches"] += 1
            i += 1

            # Pack or truncate the batch to target batch size (batch size might be variable due to sequence packing).
            if batch["input_ids"].shape[0] != cfg.dataloader.train.batch_size:
                batch, stored_batch = to_target_batch_size(batch, stored_batch, cfg.dataloader.train.batch_size)

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            if batch["input_ids"].shape[0] < cfg.dataloader.train.batch_size:
                stored_batch = batch
                continue

            # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
            # called, and the first call to .backward() outside this context manager will trigger the synchronization.
            # Accumulating manually gives more flexibility and is compatible with TPUs.
            if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    # Forward pass
                    # print("computing batch")
                    output = model(batch["input_ids"], batch.get("attention_mask", None))
                    
                    # print("OK")

                    logits = output["logits"]
                    
                    # print("logits", logits)
                    train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))
                    # print("train loss", train_loss)

                    metrics["train/CEL"] = train_loss.item()

                    entropy_loss = 0
                    if cfg.trainer.entropy_regularization_lambda > 0 :
                        all_pos_sem_attentions = output["all_pos_sem_attentions"] # listlayer, listpos/sem, [batch_size, num heads, seqlen, seqlen]
                        
                        for pos, sem, bias in all_pos_sem_attentions :
                            if bias: 
                                pos = torch.add(pos, bias)
                            reg = (1 - compute_head_entropy(sem)/torch.log(torch.tensor(2 * sem.size(-1) - 1))).detach() * compute_head_entropy(pos)
                            entropy_loss += reg.sum()
                            #print(entropy_loss)
                        
                        # print("entropy reg", entropy_loss)

                        metrics["train/local_sum_entropy_reg"] += entropy_loss.item() * batch["input_ids"].shape[0]

                    

                    
                    total_loss = train_loss + cfg.trainer.entropy_regularization_lambda * entropy_loss
                    #print("total", total_loss)
                    accelerator.backward(total_loss)


                    # Log metrics
                    metrics["train/local_total_loss"]  += total_loss.item() * batch["input_ids"].shape[0]
                    metrics["train/local_samples"] += batch["input_ids"].shape[0]
                    if "attention_mask" in batch.keys():
                        metrics["train/local_tokens"] += (batch["attention_mask"] == 0).sum().item()
                    else:
                        metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                    metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                    metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                    metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

            else:
                # Forward pass
                output = model(batch["input_ids"], batch.get("attention_mask", None))

                logits = output["logits"]
                    
                train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))
                
                entropy_loss = 0
                if cfg.trainer.entropy_regularization_lambda > 0 :
                    all_pos_sem_attentions = output["all_pos_sem_attentions"] # listlayer, listpossem, [batch_size, num heads, seqlen, seqlen]
                    
                    for pos, sem, bias in all_pos_sem_attentions :
                        if bias: 
                            pos = torch.add(pos, bias)
                        reg = (1 - compute_head_entropy(sem)/torch.log(torch.tensor(2 * sem.size(-1) - 1))).detach() * compute_head_entropy(pos)
                        entropy_loss += reg.sum()
                        #print(entropy_loss)

                    metrics["train/local_sum_entropy_reg"] += entropy_loss.item() * batch["input_ids"].shape[0]
            
                    total_loss = train_loss + cfg.trainer.entropy_regularization_lambda * entropy_loss

                else :
                    total_loss = train_loss

                # print(train_loss, entropy_loss, cfg.trainer.entropy_regularization_lambda * entropy_loss, total_loss)
                # print("BACKWARD!!!")
                accelerator.backward(total_loss)
                
                if cfg.trainer.gradient_clipping is not None and cfg.trainer.gradient_clipping > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)

                pbar.update(1)
                # Log metrics
                metrics["train/total_loss"] += total_loss.item()
                metrics["train/steps"] += 1
                metrics["train/local_samples"] += batch["input_ids"].shape[0]
                if "attention_mask" in batch.keys():
                    metrics["train/local_tokens"] += (batch["attention_mask"] == 0).sum().item()
                else:
                    metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
                metrics["train/local_sum_loss"] += train_loss.item() * (batch["labels"] != -100).sum().item()
                metrics["train/local_num_correct"] += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()

                if metrics["train/steps"] % cfg.wandb.log_interval == 0:
                    # https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.utils.safe_get_full_grad
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        metrics["train/grad_norm"] = model.get_global_grad_norm()
                        metrics["train/weight_norm"] = (
                            sum([safe_get_full_fp32_param(p).norm(2) ** 2 for p in model.parameters()]) ** 0.5
                        ).item()
                    # DDP
                    else:
                        metrics["train/grad_norm"] = (sum([p.grad.norm(2) ** 2 for p in model.parameters()]) ** 0.5).item()
                        metrics["train/weight_norm"] = (sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5).item()

                    metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    metrics.log(accelerator)

                # Save the accelerator state from the main process
                if metrics["train/steps"] % cfg.trainer.accelerate.save_steps == 0:
                    accelerator.save_state()
                    if isinstance(train_dataset, IterableDataset) :
                        torch.save(
                                train_dataset.state_dict(),
                                os.path.join(dataset_checkpoint_dir, "iterableDataset.pt"),
                            )

                # Save the pytorch model
                if metrics["train/steps"] % cfg.trainer.model.save_steps == 0:
                    if cfg.trainer.model.max_ckpt is not None:
                        # Delete checkpoints if there are too many
                        files = os.listdir(model_checkpoint_dir)
                        iterations = [int(f) for f in files if f.isdigit()]
                        iterations.sort()

                        # Remove files with the smallest iterations until the limit is met
                        while iterations is not None and len(iterations) >= cfg.trainer.model.max_ckpt:
                            file_to_remove = iterations.pop(0)
                            shutil.rmtree(os.path.join(model_checkpoint_dir, str(file_to_remove)))
                            print(
                                f"Deleted old model checkpoint {file_to_remove} due to limit " f"(max_ckpt = {cfg.trainer.model.max_ckpt})"
                            )
                    # Save the checkpoint
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        model.save_checkpoint(model_checkpoint_dir, tag=metrics["train/steps"])
                    else:
                        path = os.path.join(model_checkpoint_dir, str(metrics["train/steps"]))
                        os.makedirs(path, exist_ok=True)
                        torch.save(
                            model.state_dict(),
                            os.path.join(path, "state_dict.pt"),
                        )
                    


                if metrics["train/steps"] >= cfg.trainer.max_steps:
                    break

                # Reset the gradient
                optimizer.zero_grad()

        # Log metrics
        metrics["train/epochs"] += 1

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
