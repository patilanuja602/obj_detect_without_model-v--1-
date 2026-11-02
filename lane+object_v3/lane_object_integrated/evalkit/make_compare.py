import json, cv2, numpy as np
from viewtool import load
G=json.load(open('gt_A.json'))['frames']; BA=json.load(open('/home/claude/work/base_A.json'))['frames']; NA=json.load(open('/home/claude/work/new_A.json'))['frames']
def panel(fi, dets, title, kind, reg, s):
    f=load('A',fi); x,y,w,h=reg; c=cv2.resize(f[y:y+h,x:x+w],None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
    P=lambda px,py:(int(round((px-x)*s)),int(round((py-y)*s)))
    for v in G[str(fi)]:
        b=v['box']; top=b[1] if b[1] is not None else y
        cv2.rectangle(c,P(b[0],top),P(b[2],b[3]),(0,0,255),1)
        for q in v['contacts']: cv2.drawMarker(c,P(*q),(0,0,255),cv2.MARKER_CROSS,12,2)
    col=(255,120,0) if kind=='base' else (0,220,0)
    for d in dets[fi]['dets']:
        b=d['box']
        if b[2]<x or b[0]>x+w: continue
        cv2.rectangle(c,P(b[0],b[1]),P(b[2],b[3]),col,2)
        if d['contact']: cv2.circle(c,P(*d['contact']),5,(255,0,255),-1)
        for t in d.get('cc') or []: cv2.drawMarker(c,P(*t),(0,255,255),cv2.MARKER_TILTED_CROSS,10,2)
    cv2.rectangle(c,(0,0),(c.shape[1],22),(0,0,0),-1); cv2.putText(c,title,(6,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return c
rows=[]
for fi,reg in ((869,(1180,880,330,150)),(1049,(1200,860,380,210)),(1259,(1020,890,370,160)),(1379,(1030,880,300,160)),(1858,(1120,890,200,160))):
    s=min(3,780//reg[2]*1.0) if reg[2]<260 else 2
    s=max(2,int(780/reg[2]))
    a=panel(fi,BA,"ORIGINAL  frame %d"%fi,'base',reg,s); b=panel(fi,NA,"NEW  frame %d"%fi,'new',reg,s)
    rows.append(np.hstack([a,b]))
W=max(r.shape[1] for r in rows)
rows=[cv2.copyMakeBorder(r,0,6,0,W-r.shape[1],cv2.BORDER_CONSTANT) for r in rows]
cv2.imwrite('/home/claude/proj/lane_object_integrated/results/before_after_gt_frames.png',np.vstack(rows)); print(np.vstack(rows).shape)
