# CMT

## package requirement  
python\==3.8.13
dgl\==0.8.1
numpy\==1.21.5
pandas\==1.4.3
pytorch\==1.10.1

## data

competition website：https://ai.ppdai.com/mirror/goToMirrorDetailSix?mirrorId=28

Download the raw data to a directory, and replace the corresponding directory in Input.py

## run

Before running any method below, replace the corresponding directory for storing log file and trained model file in each python file.

### GCN

```python
python GCN.py
```

### GAT

```python
python GAT.py
```

### RGCN

```python
python RGCN.py
```

### AddGraph_homo

```python
python AddGraph/run.py
```

### AddGraph_hetero

```python
python AddGraph/run_hetero.py
```

### DCI

```python
python DCI/main.py
```

### GeniePath

```python
python GeniePath/main.py
```

#### SimpleHGN

```python
python SimpleHGN.py
```

### CMT

#### HG_Encoder

python HG_Encoder.py

#### Temporal Snapshot Sequence(TSS)

1. Get the representation for each user under different snapshots:

   ```python
   python pretrainHeteroDynamic.py
   ```

2. Run Transformer Encoder to transform the obtained user behavioral sequences into constructed tss feature:

   ```python
   python TFEncoder_tss.py
   ```

#### User Relation Sequence(URS)

1. Construct and sample the user relation sequence for each user:

   ```python
   # run user_seq_construct()
   python Input.py
   ```

2. Run Transformer Encoder to transform the obtained user relation sequence into constructed tss feature:

   ```python
   python TFEncoder_urs.py
   ```

#### Sequence Constrastive Learning

run the following scipt to acquire the sequential feautre transformed with Transformer Encoder by multi-task learning with contrastive learning.

```python
# Temporal Snapshot Sequence
python contrastive/tss.py
# User Relation Sequence
python contrastive/urs.py
```

#### Combine Together

Concat the two sequential features (tss and urs)  with raw feature as the input of graph classification model at  current timestamp.

```python
python concat/tss_cl_urs_cl.py
```

