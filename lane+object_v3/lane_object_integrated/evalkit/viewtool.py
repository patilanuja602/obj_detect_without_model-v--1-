import cv2, numpy as np, sys, glob
def load(tag, idx): return cv2.imread("frames/%s_%05d.png" % (tag, idx))
def overview(f, y0=700, y1=1130, s=0.5, label=""):
    c = f[y0:y1].copy()
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for x in range(0, f.shape[1], 100):
        X = int(x*s); cv2.line(c, (X,0), (X,c.shape[0]), (0,255,255) if x%500==0 else (90,90,90), 1)
        if x%200==0: cv2.putText(c, str(x), (X+2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    for y in range((y0//50+1)*50, y1, 50):
        Y = int((y-y0)*s); cv2.line(c, (0,Y), (c.shape[1],Y), (90,90,90), 1)
        cv2.putText(c, str(y), (2, Y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(c, label, (c.shape[1]-120, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return c
def zoom(f, x, y, w, h, s=5, boxes=(), pts=()):
    x0, y0 = max(int(x),0), max(int(y),0)
    c = f[y0:y0+h, x0:x0+w]
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for gx in range((x0//10+1)*10, x0+w, 10):
        X = (gx-x0)*s; big = gx%50==0
        cv2.line(c, (X,0), (X,c.shape[0]), (0,255,255) if big else (70,70,70), 1)
        if gx%20==0: cv2.putText(c, str(gx), (X+2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    for gy in range((y0//10+1)*10, y0+h, 10):
        Y = (gy-y0)*s; big = gy%50==0
        cv2.line(c, (0,Y), (c.shape[1],Y), (0,255,0) if big else (70,70,70), 1)
        if gy%20==0: cv2.putText(c, str(gy), (2, Y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    for (bx0,by0,bx1,by1) in boxes:
        cv2.rectangle(c, (int((bx0-x0)*s),int((by0-y0)*s)), (int((bx1-x0)*s),int((by1-y0)*s)), (0,0,255), 1)
    for (px,py) in pts:
        cv2.drawMarker(c, (int((px-x0)*s),int((py-y0)*s)), (255,0,255), cv2.MARKER_CROSS, 12, 1)
    return c
if __name__ == "__main__":
    tag = sys.argv[1]; idxs = [int(a) for a in sys.argv[2:]]
    ims = [overview(load(tag, i), label=str(i)) for i in idxs]
    cv2.imwrite("views/ov_%s_%s.png" % (tag, "_".join(map(str, idxs))), np.vstack(ims))
