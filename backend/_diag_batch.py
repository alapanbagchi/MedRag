from dotenv import load_dotenv
load_dotenv()
import os, httpx

base = os.environ["EMBEDDING_BASE_URL"].rstrip("/")
url = base + "/embeddings"
sentence = "Pulmonary arterial hypertension is a progressive disease. "
for n in [1, 32, 64, 128, 256, 512, 1000]:
    texts = [sentence * 10 for _ in range(n)]
    try:
        r = httpx.post(url, json={"model": "ncbi/MedCPT-Article-Encoder", "input": texts}, timeout=90)
    except Exception as e:
        print(n, "ERR", str(e)[:120]); break
    if r.status_code == 200:
        print(n, "OK", len(r.json().get("data", [])))
    else:
        print(n, "HTTP", r.status_code, "->", r.text[:220])
        break
