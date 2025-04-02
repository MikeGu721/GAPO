import os

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
import json
from typing import Tuple, List, Dict, Optional
import random
from sklearn.model_selection import train_test_split


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_generator_output(
    data_dir: str,
    separator: str,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    model = None,
    test_size: float = 0.2,
    batch_size: int = 16,
    max_length: int = 512,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    读取数据并使用模型生成输出
    
    Args:
        data_dir: 数据目录
        separator: 分隔符号
        max_train_samples: 训练数据上限
        max_test_samples: 测试数据上限
        model: 用于生成的模型
        test_size: 测试集比例（当没有test.jsonl时使用）
        batch_size: 批处理大小
        max_length: 生成文本的最大长度
        random_seed: 随机种子
    
    Returns:
        Tuple[List[Dict], List[Dict]]: 训练集和测试集
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    train_path = os.path.join(data_dir, 'train.jsonl')
    test_path = os.path.join(data_dir, 'test.jsonl')
    
    def read_and_process_file(file_path: str) -> List[str]:
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 获取chosen和reject中分隔符之前的内容作为prompt
                chosen_prompt = data['chosen'].split(separator)[0]+separator
                reject_prompt = data['reject'].split(separator)[0]+separator
                prompts.extend([chosen_prompt.strip(), reject_prompt.strip()])
        return prompts

    # 读取训练数据
    train_prompts = read_and_process_file(train_path)
    
    # 读取或划分测试数据
    test_prompts = []
    if os.path.exists(test_path):
        test_prompts = read_and_process_file(test_path)
    else:
        train_prompts, test_prompts = train_test_split(
            train_prompts,
            test_size=test_size,
            random_state=random_seed
        )
    
    def generate_responses(prompts: List[str], max_samples: Optional[int]) -> List[Dict]:
        """使用模型生成回复"""
        if max_samples is not None:
            prompts = prompts[:max_samples]
        
        generated_data = []
        
        # 创建数据加载器进行批处理
        dataset = [(i, p) for i, p in enumerate(prompts)]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_prompts in tqdm(dataloader, desc="Generating responses"):
                # 生成回复
                outputs = model.generate(
                    input_ids=model.tokenizer(
                        list(batch_prompts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).input_ids.to(model.device),
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id
                )
                
                # 解码生成的文本
                generated_texts = model.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                
                # 保存结果
                for prompt, generated_text in zip(batch_prompts, generated_texts):
                    generated_data.append({
                        "prompt": prompt,
                        "output": generated_text,
                        "label": 0
                    })
        
        return generated_data
    
    # 生成训练和测试数据
    generative_train_data = generate_responses(train_prompts, max_train_samples)
    generative_test_data = generate_responses(test_prompts, max_test_samples)
    
    print(f"生成的训练数据大小: {len(generative_train_data)}")
    print(f"生成的测试数据大小: {len(generative_test_data)}")
    
    return generative_train_data, generative_test_data

def mix_cls_dataset(
    data: List[Dict],
    generator_data: List[Dict],
    ratio: float,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    混合并平衡数据集
    
    Args:
        train_data: 原始训练数据
        test_data: 原始测试数据
        generator_train_data: 生成的训练数据
        generator_test_data: 生成的测试数据
        ratio: 目标比例(0标签样本占比)
        random_seed: 随机种子
    
    Returns:
        Tuple[List[Dict], List[Dict]]: 平衡后的训练集和测试集
    """
    random.seed(random_seed)
    
    def balance_dataset(
        original_data: List[Dict],
        generator_data: List[Dict],
        target_ratio: float
    ) -> List[Dict]:
        # 分离原始数据中的正负样本
        positive_samples = [x for x in original_data if x['label'] == 1]
        negative_samples = [x for x in original_data if x['label'] == 0]
        
        # 计算目标的负样本数量
        total_positive = len(positive_samples)
        target_negative = int(total_positive * target_ratio / (1 - target_ratio))
        
        # 确定需要多少生成的样本
        needed_negative = target_negative - len(negative_samples)
        
        if needed_negative > 0:
            # 需要添加生成的样本
            additional_samples = random.sample(
                generator_data,
                min(needed_negative, len(generator_data))
            )
            negative_samples.extend(additional_samples)
        else:
            # 需要减少原始负样本
            negative_samples = random.sample(negative_samples, target_negative)
        
        # 合并数据
        balanced_data = positive_samples + negative_samples
        random.shuffle(balanced_data)
        
        return balanced_data
    
    # 平衡训练集和测试集
    balanced = balance_dataset(data, generator_data, ratio)
    
    return balanced_train, balanced_test



def get_cls_dataset(
    data_dir: str,
    test_size: float = 0.2,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    '''
    从指定目录读取数据并处理成分类任务格式
    
    Args:
        data_dir: 数据目录路径
        test_size: 如果需要划分测试集，测试集占比
        max_train_samples: 控制训练集最大样本数，None表示使用全部
        max_test_samples: 控制测试集最大样本数，None表示使用全部
        random_seed: 随机种子，用于复现结果
    
    Returns:
        Tuple[List[Dict], List[Dict]]: 训练集和测试集
        每条数据格式为 {"prompt": str, "output": str, "label": int}
    '''
    random.seed(random_seed)
    
    train_path = os.path.join(data_dir, 'train.jsonl')
    test_path = os.path.join(data_dir, 'test.jsonl')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练文件不存在: {train_path}")
    
    raw_train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_train_data.append(json.loads(line.strip()))
    
    raw_test_data = []
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_test_data.append(json.loads(line.strip()))
    else:
        raw_train_data, raw_test_data = train_test_split(
            raw_train_data,
            test_size=test_size,
            random_state=random_seed
        )
    
    def convert_format(data: List[Dict]) -> List[Dict]:
        converted_data = []
        for item in data:
            converted_data.append({
                "prompt": "",
                "output": item["chosen"],
                "label": 1
            })
            converted_data.append({
                "prompt": "",
                "output": item["reject"],
                "label": 0
            })
        return converted_data
    
    train_data = convert_format(raw_train_data)
    test_data = convert_format(raw_test_data)
    
    if max_train_samples is not None:
        train_data = random.sample(train_data, min(max_train_samples, len(train_data)), shuffle=False)
    if max_test_samples is not None:
        test_data = random.sample(test_data, min(max_test_samples, len(test_data)), shuffle=False)
    
    return train_data, test_data

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
