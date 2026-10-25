"""Download the released fusion checkpoint and exact official pretrained priors."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

SEG_REV='d633b2072669ca68d8f8e309de9b52bfdbf6bf72'
SEG_BASE=f'https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024/resolve/{SEG_REV}/'
FILES=[
 ('weights/SAFIR_final.safetensors','https://github.com/eliaukyh17-droid/SAFIR/releases/download/weights-v0.2.0/SAFIR_final.safetensors','de83cc6408c9ab4800cf018c04b36ead50411da8894f7ba03e79f35517a2c1db'),
 ('pretrained/yolov8s.pt','https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt','268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d'),
 ('pretrained/segformer-b2/pytorch_model.bin',SEG_BASE+'pytorch_model.bin','584d72ace8c17c5683c198e1a4c3eda2e1026bafc02711ac5e5a4dc097bd05fe'),
 ('pretrained/segformer-b2/config.json',SEG_BASE+'config.json','3bafe3a84c87270732bb383f6ef33801d1cd441907da8cedc2c8a4986276945d'),
]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download(path,url,expected):
    if path.exists():
        if expected and digest(path)!=expected:
            raise ValueError(f'Existing file checksum mismatch: {path}')
        print(f'Present: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.part')
    print(f'Downloading {url}',flush=True)
    request=urllib.request.Request(url,headers={'User-Agent':'SAFIR-reproduction/1.0'})
    with urllib.request.urlopen(request,timeout=120) as response,temp.open('wb') as f:
        while block:=response.read(1024*1024):f.write(block)
    if expected and digest(temp)!=expected:
        raise ValueError(f'Download checksum mismatch: {temp}')
    temp.replace(path)
    print(f'Verified: {path}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--only',choices=['all','fusion','priors'],default='all')
    args=p.parse_args()
    for name,url,expected in FILES:
        if args.only=='fusion' and not name.startswith('weights/'):continue
        if args.only=='priors' and not name.startswith('pretrained/'):continue
        download(args.root/name,url,expected)


if __name__=='__main__':main()
