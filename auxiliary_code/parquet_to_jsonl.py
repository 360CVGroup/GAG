# pip install pandas pyarrow
import pandas as pd

src = "/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/dataset/additional_professional_domains/math/gsm8k/main/test-00000-of-00001.parquet"
dst = src.rsplit(".", 1)[0] + ".jsonl"

df = pd.read_parquet(src)  # 默认用 pyarrow 引擎
df.to_json(dst, orient="records", lines=True, force_ascii=False)
print(f"写入完成: {dst}")
