"""Label-blind acquisition-artifact normalization for all supplied drawings.

It intentionally does not detect/remove text or manually tune any image.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps

SIZE=128

def local_mean(a, radius=9):
    """Integral-image box mean with a global radius, not per-image tuning."""
    p=np.pad(a.astype(np.float32),radius,mode='reflect'); ii=np.pad(p.cumsum(0).cumsum(1),((1,0),(1,0)))
    k=2*radius+1
    return (ii[k:,k:]-ii[:-k,k:]-ii[k:,:-k]+ii[:-k,:-k])/(k*k)

def skeletonize(binary):
    a=(binary==0).astype(np.uint8)
    while True:
        changed=False
        for stage in (0,1):
            p2,p3,p4,p5,p6,p7,p8,p9=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            n=p2+p3+p4+p5+p6+p7+p8+p9
            tr=((p2==0)&(p3==1))+((p3==0)&(p4==1))+((p4==0)&(p5==1))+((p5==0)&(p6==1))+((p6==0)&(p7==1))+((p7==0)&(p8==1))+((p8==0)&(p9==1))+((p9==0)&(p2==1))
            guard=(p2*p4*p6==0)&(p4*p6*p8==0) if stage==0 else (p2*p4*p8==0)&(p2*p6*p8==0)
            delete=(a[1:-1,1:-1]==1)&(n>=2)&(n<=6)&(tr==1)&guard
            if delete.any(): a[1:-1,1:-1][delete]=0; changed=True
        if not changed: return np.where(a,0,255).astype(np.uint8)

def components(mask):
    """Small-image 8-connected component count; ignores <2 px specks."""
    h,w=mask.shape; seen=np.zeros_like(mask,bool); count=0
    for y,x in zip(*np.where(mask)):
        if seen[y,x]: continue
        stack=[(int(y),int(x))]; seen[y,x]=1; n=0
        while stack:
            yy,xx=stack.pop(); n+=1
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]: seen[ny,nx]=1; stack.append((ny,nx))
        count+=n>=2
    return int(count)

def remove_small(mask, minimum=4):
    """Global morphological rule for binary only; clean-gray is never erased."""
    h,w=mask.shape; seen=np.zeros_like(mask,bool); out=np.zeros_like(mask,bool)
    for y,x in zip(*np.where(mask)):
        if seen[y,x]: continue
        stack=[(int(y),int(x))]; seen[y,x]=1; pts=[]
        while stack:
            yy,xx=stack.pop(); pts.append((yy,xx))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]: seen[ny,nx]=1; stack.append((ny,nx))
        if len(pts)>=minimum:
            for yy,xx in pts: out[yy,xx]=1
    return out

def metrics(stage, gray, mask, skeleton):
    bg=gray[~mask]; ys,xs=np.where(mask)
    bbox=0 if not len(xs) else ((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))/gray.size
    edges=np.abs(np.diff(gray.astype(float),axis=0)).mean()+np.abs(np.diff(gray.astype(float),axis=1)).mean()
    return {f'{stage}_background_mean':float(bg.mean()) if len(bg) else np.nan,f'{stage}_background_std':float(bg.std()) if len(bg) else np.nan,
            f'{stage}_foreground_ratio':float(mask.mean()),f'{stage}_edge_pixels':float(edges),f'{stage}_connected_components':components(mask),
            f'{stage}_skeleton_length':int((skeleton==0).sum()),f'{stage}_line_bbox_coverage':float(bbox)}

def process(path):
    # Work at max 512 px, so illumination radius corresponds to acquisition scale, not camera resolution.
    raw=ImageOps.exif_transpose(Image.open(path)).convert('RGB'); raw.thumbnail((512,512)); a=np.asarray(raw).astype(np.float32)
    # Scheme A: per-channel blurred background ratio removes paper colour and illumination before grayscale.
    bg=np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=35))).astype(np.float32)
    corrected=np.clip(a/np.maximum(bg,8)*255,0,255)
    gray=(.2126*corrected[...,0]+.7152*corrected[...,1]+.0722*corrected[...,2]).astype(np.uint8)
    # Local darkness is a label-blind adaptive rule. A low fixed floor (3/255) protects faint pencils.
    mean=local_mean(gray,9); darkness=mean-gray
    local_dev=local_mean(np.abs(gray-mean),9)
    mask=darkness>np.maximum(3.0,0.35*local_dev)
    # one-pixel dilation preserves stroke width/topology without filling intersections.
    padded=np.pad(mask,1); mask=np.logical_or.reduce([padded[i:i+gray.shape[0],j:j+gray.shape[1]] for i in range(3) for j in range(3)])
    # Keep continuous corrected line intensity only in the mask. Background and all padding are exactly white.
    clean=np.where(mask,gray,255).astype(np.uint8)
    # Aspect-ratio preserving downsampling; no signal-dependent crop.
    im=Image.fromarray(clean); im.thumbnail((SIZE,SIZE),Image.Resampling.LANCZOS); canvas=Image.new('L',(SIZE,SIZE),255); canvas.paste(im,((SIZE-im.width)//2,(SIZE-im.height)//2)); clean=np.asarray(canvas)
    # Binary/skeleton are ablations: use a stricter global threshold and remove tiny isolated specks.
    # The continuous clean-gray source remains untouched so faint strokes are retained for the main input.
    clean_mask=clean<254; binary_mask=remove_small(clean<235,minimum=4); binary=np.where(binary_mask,0,255).astype(np.uint8); skel=skeletonize(binary)
    # Reference metrics are calculated at equal 128x128 dimensions.
    rawgray=ImageOps.grayscale(raw); rawgray.thumbnail((SIZE,SIZE),Image.Resampling.LANCZOS); rawcanvas=Image.new('L',(SIZE,SIZE),255); rawcanvas.paste(rawgray,((SIZE-rawgray.width)//2,(SIZE-rawgray.height)//2)); raw128=np.asarray(rawcanvas); rawmask=raw128<(local_mean(raw128,9)-3); rawskel=skeletonize(np.where(rawmask,0,255).astype(np.uint8))
    return raw,clean,binary,skel,metrics('before',raw128,rawmask,rawskel),metrics('after',clean,clean_mask,skel)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--image-dir',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
    source=Path(a.image_dir); output=Path(a.output_dir)
    for name in ('clean_gray','binary','skeleton'): (output/name).mkdir(parents=True,exist_ok=True)
    images=sorted((x for x in source.iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}),key=lambda x:x.name)
    rows=[]
    for path in images:
        _,clean,binary,skel,before,after=process(path); name=path.stem+'.png'; Image.fromarray(clean).save(output/'clean_gray'/name); Image.fromarray(binary).save(output/'binary'/name); Image.fromarray(skel).save(output/'skeleton'/name)
        rows.append({'image_file':path.name,**before,**after})
    pd.DataFrame(rows).to_csv(output/'preprocessing_metrics.csv',index=False,encoding='utf-8-sig'); print(f'Processed {len(images)} images with one label-blind rule.')
if __name__=='__main__': main()
