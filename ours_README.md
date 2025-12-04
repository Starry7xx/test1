//equiformer_v2-main模型主要是用于生成数据集中的eq_emb的数据
## 准备工作
重要！！！进入equi_v2环境 
```bash
conda activate equi_v2
```
## 数据生成
1.先生成LMDB文件
有geom和castep结果文件的：
```bash
python database/Lmdb/Lmdb_normal.py
```
只有castep，没有geom文件的：
```bash
python database/Lmdb/Lmdb_no_geom.py
```
有castep和geom文件，但是geom文件不完整的：
```bash
python database/Lmdb/Lmdb_presentation_error.py
```
检查生成的LMDB文件是否正确：
```bash
python database/check_Lmdb.py
```

2.生成3200维的迁入向量npz文件/L_4和L_6生成的维度不同
```bash
python extract_embeddings_L_4.py
```
修改路径：输出/输入
DATA_PATH = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/equiformer_v2-main/database/Lmdb/Cd/O3.lmdb"
OUTPUT_FILENAME = "Cd_O3_embeddings_3200.npz"

3.生成权重文件
```bash
python database/get_clean_emb.py
```
修改路径：输出/输入
npz_path = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/equiformer_v2-main/database/npz/Cd_O3_embeddings_3200.npz"
torch.save(eq_emb, "Cd_O3_eq_emb.pt")

4：生存csv文件
```bash
python database/save_to_csv.py
```
修改路径：输出/输入
pt_path = "Cd_O3_eq_emb.pt"
csv_path = os.path.join(OUTPUT_DIR, "Cd_O3_eq_emb.csv")//其他输出路径可以不用管
