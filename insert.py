import pandas as pd
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from text2vec import SentenceModel


df = pd.read_csv('news_collection.csv')
id_answer = df.set_index('title')['desc'].to_dict()

model = SentenceModel('shibing624/text2vec-base-chinese')
connections.connect(host='192.168.52.5', port='19530')


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', max_length=500, is_primary=True,
                    auto_id=True),
        FieldSchema(name='question', dtype=DataType.VARCHAR, description='question content', max_length=512),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim),
        FieldSchema(name='answer', dtype=DataType.VARCHAR, description='answer content', max_length=512),
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


collection = create_milvus_collection('question_news', 768)
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

pbar = tqdm(total=len(df.index))
count = 0
for rid, row in df.iterrows():
    count += 1
    title, d, _, _, _, _ = row.values
    
    if isinstance(title, str) and len(title) > 512:
        title = title[:169]
    if isinstance(d, str) and len(d) > 168:
        d = d[:168]
    if not isinstance(d, str):
        print(f"{d}不符合条件，跳过")
        continue
    
    collection.insert([[title], [model.encode(title)], [d]])
    pbar.update(1)
print('Total number of inserted data is {}.'.format(collection.num_entities))
