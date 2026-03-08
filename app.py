from flask import Flask, request, jsonify, render_template_string
import os
import pickle
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load CIFAR-10 data at startup
DATA_DIR = os.path.join(os.path.dirname(__file__), "cifar-10-batches-py")
TRAIN_FILES = [os.path.join(DATA_DIR, f"data_batch_{i}") for i in range(1, 6)]
TEST_FILE = os.path.join(DATA_DIR, "test_batch")
META_FILE = os.path.join(DATA_DIR, "batches.meta")

def _load_pickle(path):
    with open(path, 'rb') as f:
        try:
            return pickle.load(f, encoding='latin1')
        except TypeError:
            return pickle.load(f)

print("Loading CIFAR-10 data...")
train_list = []
train_labels_list = []
for tf in TRAIN_FILES:
    if not os.path.exists(tf):
        raise FileNotFoundError(f"Missing file: {tf}")
    batch = _load_pickle(tf)
    if 'data' in batch:
      data = batch['data']
    else:
      data = batch[b'data']
    if 'labels' in batch:
      labels = batch['labels']
    else:
      labels = batch[b'labels']
    train_list.append(np.array(data, dtype=np.uint8))
    train_labels_list.extend(list(labels))

train_data = np.vstack(train_list)
train_labels = np.array(train_labels_list, dtype=np.int64)

test_batch = _load_pickle(TEST_FILE)
if 'data' in test_batch:
  test_data = np.array(test_batch['data'], dtype=np.uint8)
else:
  test_data = np.array(test_batch[b'data'], dtype=np.uint8)
if 'labels' in test_batch:
  test_labels = np.array(test_batch['labels'], dtype=np.int64)
else:
  test_labels = np.array(test_batch[b'labels'], dtype=np.int64)

meta = _load_pickle(META_FILE)
if 'label_names' in meta:
  label_names = meta['label_names']
else:
  label_names = meta[b'label_names']
label_names = [ln.decode('utf-8') if isinstance(ln, bytes) else str(ln) for ln in label_names]

print(f"Loaded train {train_data.shape}, test {test_data.shape}")

def cifar_row_to_image(row):
    # row: (3072,) uint8 with R..G..B.. ordering
    arr = row.reshape(3, 32, 32).transpose(1, 2, 0)
    return arr

def image_to_base64_png(img_array, scale=3):
    # img_array: HxWx3 uint8
    im = Image.fromarray(img_array)
    new_size = (img_array.shape[1] * scale, img_array.shape[0] * scale)
    im = im.resize(new_size, Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    b = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b}"

@app.route('/')
def index():
    # Single-file template using render_template_string
    return render_template_string('''
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>CIFAR-10 k-NN</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono&family=DM+Sans:wght@400;700&family=Playfair+Display:ital,wght@0,400;1,700;1,900&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#0d0810; --card:#130d17; --card-border:#2a1a35; --rose:#ff3f8e; --lilac:#c084fc; --blush:#ff85b3; --mint:#06d6a0; --accent-grad:linear-gradient(90deg,var(--rose),var(--lilac));
    }
    html,body{height:100%;margin:0;background:var(--bg);font-family:'DM Sans',system-ui,Arial;color:#ddd;}
    body{
      background-image: radial-gradient(circle at 10% 10%, rgba(192,132,252,0.08), transparent 10%), radial-gradient(circle at 90% 90%, rgba(255,65,150,0.06), transparent 10%), linear-gradient(135deg, rgba(255,255,255,0.01), transparent 40%);
      background-color: var(--bg);
      background-blend-mode: screen;
      padding:32px;
      -webkit-font-smoothing:antialiased;
    }
    .container{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:24px;align-items:center;padding:0 24px}
    .header{margin-bottom:48px;text-align:left;box-sizing:border-box;overflow:visible;width:100%}
    .eyebrow{font-family:'DM Mono';color:var(--rose);font-size:12px;letter-spacing:1px;text-align:left;width:100%;box-sizing:border-box}
    .title-wrap{display:flex;flex-direction:column;gap:6px;align-items:flex-start;overflow:visible}
    /* decorative line removed as requested */
    .title-main{margin:0;font-family:'Playfair Display';font-style:normal;font-weight:900;font-size:3.5rem;color:#ffffff}
    .title-sub{margin:0;font-family:'Playfair Display';font-style:italic;font-weight:900;font-size:4.5rem;line-height:1.15;padding-bottom:28px;background:linear-gradient(90deg,var(--rose),var(--lilac));-webkit-background-clip:text;background-clip:text;color:transparent;overflow:visible;display:block}
    .card{background:var(--card);border:1px solid var(--card-border);border-radius:24px;padding:18px;box-shadow:0 6px 30px rgba(0,0,0,0.6);position:relative;overflow:hidden}
    .card::before{content:"";position:absolute;left:0;right:0;height:6px;top:0;background:linear-gradient(90deg,var(--rose),var(--lilac));opacity:0.12}
    .card{width:100%}
    .controls{display:flex;flex-direction:column;gap:12px}
    .metric-row{display:flex;gap:12px}
    .metric{flex:1;padding:18px;border-radius:16px;border:1px solid #2a1a35;background:#180916;cursor:pointer;display:flex;flex-direction:column;align-items:flex-start;gap:6px}
    .metric h2{margin:0;font-family:'Playfair Display';font-style:italic;font-weight:900;font-size:36px}
    .metric p{margin:0;font-family:'DM Mono';font-size:13px;color:#ccc}
    .inputs{display:flex;gap:12px}
    input[type=number]{flex:1;padding:12px;border-radius:12px;border:1px solid #2a1a35;background:transparent;color:#fff;font-family:'DM Mono'}
    .submit{width:100%;padding:14px;border-radius:14px;border:0;background:var(--accent-grad);color:#111;font-family:'Playfair Display';font-style:italic;font-weight:900;cursor:pointer}
    .result{display:none;flex-direction:column;gap:12px}
    .verdict{font-family:'Playfair Display';font-style:italic;font-weight:900;font-size:42px}
    .verdict.correct{color:var(--mint)}
    .verdict.wrong{color:var(--rose)}
    .query-row{display:flex;gap:12px;align-items:center}
    .query-img{width:96px;height:96px;image-rendering:pixelated;border-radius:8px;position:relative}
    .spin-ring{width:120px;height:120px;border-radius:50%;border:4px dashed rgba(255,255,255,0.06);position:absolute;left:-12px;top:-12px;animation:spin 6s linear infinite}
    @keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
    .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
    .stat{background:transparent;padding:10px;border-radius:10px;text-align:center}
    .stat .val{font-family:'Playfair Display';font-style:italic;font-weight:900;font-size:20px}
    .votes{display:flex;flex-direction:column;gap:6px}
    .vote-row{display:flex;align-items:center;gap:8px}
    .vote-bar{flex:1;height:12px;border-radius:8px;background:#221222;overflow:hidden}
    .vote-fill{height:100%;background:linear-gradient(90deg,var(--rose),var(--lilac));}
    .neighbors{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;width:100%}
    .neighbor{background:#0f0b10;padding:8px;border-radius:10px;text-align:center;border-top:4px solid transparent}
    .neighbor.top3{border-top:4px solid;}
    .neighbor img{width:100%;image-rendering:pixelated;border-radius:6px}
    .muted{opacity:0.7}
    .fadeUp{animation:fadeUp 400ms ease both}
    @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
    .small-label{font-family:'DM Mono';font-size:12px;color:#b7a8b6;letter-spacing:1px;text-transform:uppercase}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div style="display:flex;flex-direction:column;gap:6px">
        <div class="eyebrow">Ödev-1 · Derin Sinir Ağları</div>
        <div class="title-wrap">
          <div class="title-main">CIFAR-10</div>
          <div class="title-sub">k-NN</div>
          <div class="small-label" style="margin-top:8px">k-En Yakın Komşu · Görüntü Sınıflandırıcı</div>
        </div>
      </div>
    </div>

    <div class="card controls card-left">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div class="small-label" style="color:#b79aa8">✦  01 · Parametreler</div>
      </div>
      <div class="metric-row">
        <div class="metric" id="btn-l1" data-val="1">
          <h2 style="font-family:'Playfair Display';font-style:italic;font-weight:900;font-size:44px;margin:0">L1</h2>
          <div style="font-family:'DM Mono';font-size:14px;margin-top:6px">Manhattan</div>
          <div style="font-family:'DM Mono';font-size:12px;margin-top:4px">Σ |xᵢ − yᵢ|</div>
        </div>
        <div class="metric" id="btn-l2" data-val="2">
          <h2 style="font-family:'Playfair Display';font-style:italic;font-weight:900;font-size:44px;margin:0">L2</h2>
          <div style="font-family:'DM Mono';font-size:14px;margin-top:6px">Öklid</div>
          <div style="font-family:'DM Mono';font-size:12px;margin-top:4px">√Σ (xᵢ − yᵢ)²</div>
        </div>
      </div>

      <div class="inputs">
        <div style="flex:1;display:flex;flex-direction:column;gap:6px">
          <div class="small-label">K DEĞERİ</div>
          <input type="number" id="kval" placeholder="k değeri" value="5" min="1">
        </div>
        <div style="flex:1;display:flex;flex-direction:column;gap:6px">
          <div class="small-label">TEST İNDEX  (0 – 9999)</div>
          <input type="number" id="tidx" placeholder="test index (0-9999)" value="0" min="0">
        </div>
      </div>
      <button class="submit" id="submitBtn">Sınıflandır →</button>
    </div>

    <div class="card result" id="resultCard">
      <div class="small-label">✦  02 · Sonuç</div>
      <div class="verdict" id="verdictText">—</div>
      <div class="query-row">
        <div style="position:relative;display:inline-block">
          <div class="spin-ring"></div>
          <img id="queryImg" class="query-img" src="" alt="query">
        </div>
        <div>
          <div style="font-family:Playfair Display;font-style:italic;font-weight:900;font-size:28px" id="predClass">—</div>
          <div style="font-family:DM Mono;color:#aaa" id="trueClass">—</div>
        </div>
      </div>
      <div class="stats">
        <div class="stat"><div class="val" id="statMetric">—</div><div class="muted" style="font-family:DM Mono">Metrik</div></div>
        <div class="stat"><div class="val" id="statK">—</div><div class="muted" style="font-family:DM Mono">k Değeri</div></div>
        <div class="stat"><div class="val" id="statMin">—</div><div class="muted" style="font-family:DM Mono">Min. Mesafe</div></div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
        <div class="small-label">OY DAĞILIMI</div>
        <div style="flex:1;height:1px;background:#2a1a35"></div>
      </div>
      <div class="votes" id="votes"></div>
      <div style="display:flex;align-items:center;margin-top:8px;gap:8px">
        <div class="small-label">EN YAKIN K KOMŞU</div>
        <div style="flex:1;height:1px;background:#2a1a35"></div>
      </div>
      <div class="neighbors" id="neighbors"></div>
    </div>

  </div>

  <script>
    const btnL1 = document.getElementById('btn-l1');
    const btnL2 = document.getElementById('btn-l2');
    let metric = 1;
    function setMetric(m){metric=m;btnL1.style.borderColor=(m===1?'var(--rose)':'#2a1a35');btnL2.style.borderColor=(m===2?'var(--rose)':'#2a1a35');}
    btnL1.addEventListener('click',()=>setMetric(1));
    btnL2.addEventListener('click',()=>setMetric(2));
    setMetric(1);

    document.getElementById('submitBtn').addEventListener('click',async ()=>{
      const k = parseInt(document.getElementById('kval').value||'5');
      const tidx = parseInt(document.getElementById('tidx').value||'0');
      const payload = {metric:metric,k:k,test_index:tidx};
      document.getElementById('submitBtn').disabled = true;
      const res = await fetch('/classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const j = await res.json();
      document.getElementById('submitBtn').disabled = false;
      showResult(j);
    });

    function showResult(j){
      document.getElementById('resultCard').style.display='flex';
      const verdict = document.getElementById('verdictText');
      verdict.textContent = j.verdict_text;
      verdict.className = 'verdict ' + (j.correct? 'correct':'wrong');
      document.getElementById('queryImg').src = j.query_image;
      document.getElementById('predClass').textContent = j.pred_name;
      document.getElementById('trueClass').textContent = j.true_name;
      document.getElementById('statMetric').textContent = j.metric_name;
      document.getElementById('statK').textContent = j.k;
      document.getElementById('statMin').textContent = j.min_distance.toFixed(3);

      // votes
      const votesDiv = document.getElementById('votes'); votesDiv.innerHTML='';
      j.votes.forEach(v=>{
        const row = document.createElement('div'); row.className='vote-row';
        const label = document.createElement('div'); label.style.width='140px'; label.style.fontFamily='DM Mono'; label.textContent=v.name;
        const bar = document.createElement('div'); bar.className='vote-bar';
        const fill = document.createElement('div'); fill.className='vote-fill'; fill.style.width=(v.pct*100)+'%';
        if(v.winner){fill.style.background='linear-gradient(90deg,var(--rose),var(--lilac))'} else {fill.style.background='#2a1822'}
        bar.appendChild(fill);
        const cnt = document.createElement('div'); cnt.style.width='60px'; cnt.style.fontFamily='DM Mono'; cnt.textContent=v.count;
        row.appendChild(label); row.appendChild(bar); row.appendChild(cnt);
        votesDiv.appendChild(row);
      });

      // neighbors
      const neigh = document.getElementById('neighbors'); neigh.innerHTML='';
      j.neighbors.forEach((n,idx)=>{
        const card = document.createElement('div'); card.className='neighbor';
        if(idx<3) card.classList.add('top3');
        const img = document.createElement('img'); img.src = n.image; card.appendChild(img);
        const t = document.createElement('div'); t.style.fontFamily='DM Mono'; t.textContent = `#${idx+1} ${n.class_name}`; card.appendChild(t);
        const d = document.createElement('div'); d.style.fontFamily='DM Mono'; d.textContent = Number(n.distance).toLocaleString(); card.appendChild(d);
        neigh.appendChild(card);
      });
    }
  </script>
</body>
</html>
    ''')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    metric = int(data.get('metric', 1))
    k = int(data.get('k', 5))
    test_index = int(data.get('test_index', 0))

    # prepare arrays
    train_float = train_data.astype(np.float32)
    query = test_data[test_index].astype(np.float32)

    if metric == 1:
        distances = np.sum(np.abs(train_float - query), axis=1)
        metric_name = 'L1 (Manhattan)'
    else:
        distances = np.sqrt(np.sum((train_float - query) ** 2, axis=1))
        metric_name = 'L2 (Euclidean)'

    min_distance = float(np.min(distances))
    nearest = np.argsort(distances)[:k]
    neighbor_labels = train_labels[nearest]
    vote_counts = np.bincount(neighbor_labels, minlength=len(label_names))

    pred_label = int(np.argmax(vote_counts))
    true_label = int(test_labels[test_index])
    pred_name = label_names[pred_label]
    true_name = label_names[true_label]
    correct = (pred_label == true_label)

    # prepare images
    query_img = image_to_base64_png(cifar_row_to_image(test_data[test_index]), scale=3)
    neighbors_out = []
    for ni, idx in enumerate(nearest):
      arr = cifar_row_to_image(train_data[idx])
      # round distance to nearest integer for display
      dist_int = int(round(float(distances[idx])))
      neighbors_out.append({
        'train_idx': int(idx),
        'distance': dist_int,
        'label': int(train_labels[idx]),
        'class_name': label_names[int(train_labels[idx])],
        'image': image_to_base64_png(arr, scale=2)
      })

    total_votes = int(np.sum(vote_counts))
    votes_list = []
    for i, cnt in enumerate(vote_counts):
        if cnt == 0:
            continue
        votes_list.append({'name': label_names[i], 'count': int(cnt), 'pct': float(cnt) / max(1, total_votes), 'winner': (i == pred_label)})
    # sort descending
    votes_list = sorted(votes_list, key=lambda x: -x['count'])

    result = {
        'metric': metric,
        'metric_name': metric_name,
        'k': k,
        'test_index': test_index,
        'min_distance': min_distance,
        'pred_label': pred_label,
        'true_label': true_label,
        'pred_name': pred_name,
        'true_name': true_name,
        'correct': correct,
        'verdict_text': 'Doğru ✓' if correct else 'Yanlış ✗',
        'query_image': query_img,
        'neighbors': neighbors_out,
        'votes': votes_list
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
