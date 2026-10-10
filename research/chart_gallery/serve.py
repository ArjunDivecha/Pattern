"""Read-only chart gallery for the completed five-bucket seed-101 model."""
from pathlib import Path
import sys,json,io
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import pandas as pd,numpy as np
from PIL import Image
RUN=ROOT/'runs/next_open_five_class_sample101_seed101_20260918'
S=pd.read_parquet(RUN/'I20_R5_next_open_five_class/seed_101/scores.parquet')
IMAGES=np.load(RUN/'images.npy',mmap_mode='r');ALLOWED=set(S.sample_id.astype(int))
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  u=urlparse(self.path)
  try:
   if u.path=='/':body=(Path(__file__).with_name('index.html')).read_bytes();mime='text/html'
   elif u.path=='/api':
    q=parse_qs(u.query);threshold=float(q.get('p',['0.30'])[0]);offset=max(0,int(q.get('offset',['0'])[0]));mode=q.get('mode',['top'])[0]
    t=S[S.prob_class_5>=threshold]
    t=t.sample(frac=1,random_state=71) if mode=='random' else t.sort_values(['prob_class_5','date','security'],ascending=[False,True,True])
    records=[]
    for r in t.iloc[offset:offset+48].itertuples():records.append({'id':int(r.sample_id),'stock':r.security,'date':str(r.date.date()),'p':r.prob_class_5,'down':r.prob_class_1,'return':r.forward_return if np.isfinite(r.forward_return) else None})
    body=json.dumps({'count':len(t),'records':records,'total':len(S),'max_probability':float(S.prob_class_5.max())},allow_nan=False).encode();mime='application/json'
   elif u.path.startswith('/image/'):
    k=int(u.path.split('/')[-1]);assert k in ALLOWED
    b=io.BytesIO();Image.fromarray(IMAGES[k,0]).save(b,format='PNG');body=b.getvalue();mime='image/png'
   else:self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  except (ValueError,AssertionError):self.send_error(400)
 def log_message(self,*args):pass
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
