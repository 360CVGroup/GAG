import pickle
import random
import torch

def load_data(data_path):
    """逐条读取 pickle 文件中的数据"""
    data = []
    with open(data_path, "rb") as f:
        while True:
            try:
                item = pickle.load(f)
                data.append(item)
            except EOFError:
                break
    return data


if __name__ == "__main__":
    DATA_PATH = "/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/saves/dev_dataset_background_embedding/benchmark_data_with_id_with_answer_embeddding_length_less_than_1000.pkl"

    all_data = load_data(DATA_PATH)
    print(f"✅ 共加载 {len(all_data)} 条数据")

    # 随机顺序选取 5 条
    random.shuffle(all_data)
    sample_data = all_data[:5]

    for i, item in enumerate(sample_data, 1):
        print(f"\n--- Sample {i} ---")
        print(item)  # 原样输出整个字典
        # emb = item["last_layer_answer_tokens_embedding"]
        # if isinstance(emb, torch.Tensor):
        #     # print("last_layer_answer_tokens_embedding shape:", tuple(emb.shape))
        #     print("last_layer_answer_tokens_embedding shape:", emb.shape)
        # else:
        #     print("last_layer_answer_tokens_embedding is not a torch.Tensor")
