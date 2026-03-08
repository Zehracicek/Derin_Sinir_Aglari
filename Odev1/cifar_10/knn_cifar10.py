import os
import pickle
import sys
import numpy as np

# ── STEP 1: LOAD DATA ──
print("=" * 50)
print("STEP 1: LOAD DATA - Starting")

data_folder = os.path.join("cifar-10-batches-py")
train_files = [os.path.join(data_folder, f"data_batch_{i}") for i in range(1, 6)]
test_file = os.path.join(data_folder, "test_batch")
meta_file = os.path.join(data_folder, "batches.meta")

train_data_list = []
train_labels_list = []

for idx, tf in enumerate(train_files, 1):
    print(f"Loading training batch {idx}: {tf}")
    if not os.path.exists(tf):
        print(f"Error: file not found: {tf}")
        sys.exit(1)
    with open(tf, 'rb') as f:
        try:
            batch = pickle.load(f, encoding='latin1')
        except TypeError:
            batch = pickle.load(f)

    # handle possible key types (str or bytes)
    if 'data' in batch:
        data = batch['data']
    elif b'data' in batch:
        data = batch[b'data']
    else:
        raise KeyError('data key not found in batch')

    if 'labels' in batch:
        labels = batch['labels']
    elif b'labels' in batch:
        labels = batch[b'labels']
    else:
        raise KeyError('labels key not found in batch')

    train_data_list.append(np.array(data))
    train_labels_list.extend(list(labels))

print("Concatenating training batches...")
train_data = np.vstack(train_data_list)
train_labels = np.array(train_labels_list, dtype=np.int64)

print(f"Training data shape: {train_data.shape}")
print(f"Training labels shape: {train_labels.shape}")

print(f"Loading test batch: {test_file}")
if not os.path.exists(test_file):
    print(f"Error: file not found: {test_file}")
    sys.exit(1)
with open(test_file, 'rb') as f:
    try:
        test_batch = pickle.load(f, encoding='latin1')
    except TypeError:
        test_batch = pickle.load(f)

if 'data' in test_batch:
    test_data = np.array(test_batch['data'])
    test_labels = np.array(test_batch['labels'], dtype=np.int64)
elif b'data' in test_batch:
    test_data = np.array(test_batch[b'data'])
    test_labels = np.array(test_batch[b'labels'], dtype=np.int64)
else:
    raise KeyError('data key not found in test batch')

print(f"Test data shape: {test_data.shape}")
print(f"Test labels shape: {test_labels.shape}")

print(f"Loading meta file: {meta_file}")
if not os.path.exists(meta_file):
    print(f"Error: file not found: {meta_file}")
    sys.exit(1)
with open(meta_file, 'rb') as f:
    try:
        meta = pickle.load(f, encoding='latin1')
    except TypeError:
        meta = pickle.load(f)

if 'label_names' in meta:
    label_names = meta['label_names']
elif b'label_names' in meta:
    label_names = meta[b'label_names']
else:
    # fallback: create numeric names
    label_names = [str(i) for i in range(10)]

# ensure label names are strings
label_names = [ln.decode('utf-8') if isinstance(ln, bytes) else str(ln) for ln in label_names]

print("STEP 1: LOAD DATA - Done")
print("=" * 50)

# ── STEP 2: USER INPUT: METRIC ──
print("STEP 2: USER INPUT: METRIC - Starting")
print("Choosing distance metric. Enter 1 for L1 (Manhattan) or 2 for L2 (Euclidean).")
metric = None
while True:
    choice = input("Metric (1=L1, 2=L2): ").strip()
    if choice in ('1', '2'):
        metric = int(choice)
        break
    print("Invalid input. Please enter 1 or 2.")

if metric == 1:
    metric_name = 'L1 (Manhattan)'
else:
    metric_name = 'L2 (Euclidean)'

print(f"Selected metric: {metric_name}")
print("STEP 2: USER INPUT: METRIC - Done")
print("=" * 50)

# ── STEP 3: USER INPUT: K ──
print("STEP 3: USER INPUT: K - Starting")
max_k = train_data.shape[0]
while True:
    kval = input(f"Enter k (positive integer, max {max_k}): ").strip()
    try:
        k = int(kval)
        if k <= 0:
            print("k must be positive.")
            continue
        if k > max_k:
            print(f"k cannot exceed number of training samples ({max_k}).")
            continue
        break
    except ValueError:
        print("Invalid input. Enter a positive integer for k.")

print(f"Chosen k = {k}")
print("STEP 3: USER INPUT: K - Done")
print("=" * 50)

# ── STEP 4: USER INPUT: TEST INDEX ──
print("STEP 4: USER INPUT: TEST INDEX - Starting")
while True:
    tidx = input("Enter test image index (0-9999): ").strip()
    try:
        test_index = int(tidx)
        if test_index < 0 or test_index >= test_data.shape[0]:
            print(f"Index out of range. Enter between 0 and {test_data.shape[0]-1}.")
            continue
        break
    except ValueError:
        print("Invalid input. Enter an integer between 0 and 9999.")

print(f"Selected test index = {test_index}")
print("STEP 4: USER INPUT: TEST INDEX - Done")
print("=" * 50)

# ── STEP 5: SHOW QUERY IMAGE INFO ──
print("STEP 5: SHOW QUERY IMAGE INFO - Starting")
true_label = int(test_labels[test_index])
true_class_name = label_names[true_label] if true_label < len(label_names) else str(true_label)
print(f"Test image index: {test_index}")
print(f"True label index: {true_label}")
print(f"True class name: {true_class_name}")
print("STEP 5: SHOW QUERY IMAGE INFO - Done")
print("=" * 50)

# ── STEP 6: COMPUTE DISTANCES ──
print("STEP 6: COMPUTE DISTANCES - Starting")
print(f"Computing {metric_name} distances (vectorized, no loops)...")

# convert to float32 to reduce memory and speed up computations
train_float = train_data.astype(np.float32)
query = test_data[test_index].astype(np.float32)

if metric == 1:
    # L1 (Manhattan)
    distances = np.sum(np.abs(train_float - query), axis=1)
else:
    # L2 (Euclidean)
    distances = np.sqrt(np.sum((train_float - query) ** 2, axis=1))

print("Distance computation complete.")
print(f"Min distance: {distances.min()}")
print(f"Max distance: {distances.max()}")
print("STEP 6: COMPUTE DISTANCES - Done")
print("=" * 50)

# ── STEP 7: FIND K NEAREST NEIGHBORS ──
print("STEP 7: FIND K NEAREST NEIGHBORS - Starting")
print(f"Finding {k} nearest neighbors...")
nearest_idxs = np.argsort(distances)[:k]

print("Rank | TrainIdx | Distance       | Class")
print("-----+----------+----------------+----------------")
for rank, tid in enumerate(nearest_idxs, start=1):
    dist = distances[tid]
    lbl = int(train_labels[tid])
    cname = label_names[lbl] if lbl < len(label_names) else str(lbl)
    print(f"{rank:4d} | {tid:8d} | {dist:14.6f} | {cname}")

print("STEP 7: FIND K NEAREST NEIGHBORS - Done")
print("=" * 50)

# ── STEP 8: MAJORITY VOTE ──
print("STEP 8: MAJORITY VOTE - Starting")
neighbor_labels = train_labels[nearest_idxs]
vote_counts = np.bincount(neighbor_labels, minlength=len(label_names))

# prepare sorted vote table
vote_indices = np.argsort(-vote_counts)
print("Class Name | LabelIdx | VoteCount")
print("-----------+----------+----------")
for vi in vote_indices:
    if vote_counts[vi] == 0:
        continue
    name = label_names[vi] if vi < len(label_names) else str(vi)
    print(f"{name:10s} | {vi:8d} | {vote_counts[vi]:8d}")

print("STEP 8: MAJORITY VOTE - Done")
print("=" * 50)

# ── STEP 9: FINAL PREDICTION ──
print("STEP 9: FINAL PREDICTION - Starting")
pred_label = int(np.argmax(vote_counts))
pred_name = label_names[pred_label] if pred_label < len(label_names) else str(pred_label)

print(f"Predicted class: {pred_name} (label {pred_label})")
print(f"True class:      {true_class_name} (label {true_label})")
if pred_label == true_label:
    print("Result: CORRECT")
else:
    print("Result: WRONG")

print("STEP 9: FINAL PREDICTION - Done")
print("=" * 50)

print("All steps completed. Exiting.")
