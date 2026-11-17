"""Render saved Sun Compass predictions; no inference or source mutation."""
import csv, hashlib, json, math, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parent
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, type=Path)
parser.add_argument('--predictions', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--font', default='/System/Library/Fonts/Supplemental/Arial.ttf', type=str)
parser.add_argument('--bold', default='/System/Library/Fonts/Supplemental/Arial Bold.ttf', type=str)
args = parser.parse_args()
DATA = args.data
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
FONT = args.font
BOLD = args.bold
def font(n,b=False): return ImageFont.truetype(BOLD if b else FONT,n)
BG='#0c1522'; PANEL='#152335'; FG='#eef3fa'; MUTED='#afbed0'; GOLD='#ffd15c'
CAMS=['left-forward','front-forward','right-forward','left-backward','right-backward']
COLORS=['#58d9ca','#ffd15c','#8db5ff','#f19fc9','#cbabff']
POS=[(12,108),(536,108),(1060,108),(12,588),(1060,588)]
W,H=1584,1132
wrap=lambda a:(a+180)%360-180
rows=list(csv.DictReader(args.predictions.open()))
checks=[]
def endpoint(cx,cy,r,a):
 a=math.radians(a);return cx-r*math.sin(a),cy-r*math.cos(a)
def arrow(d,cx,cy,r,a,color,width=4):
 ex,ey=endpoint(cx,cy,r,a);ux=(ex-cx)/r;uy=(ey-cy)/r
 d.line((cx,cy,ex,ey),fill=color,width=width)
 d.polygon([(ex,ey),(ex-11*ux+5*uy,ey-11*uy-5*ux),(ex-11*ux-5*uy,ey-11*uy+5*ux)],fill=color)
def dial(d,cx,cy,r):
 d.ellipse((cx-r,cy-r,cx+r,cy+r),outline='#647890',width=2)
 for a,l in [(0,'F'),(90,'L'),(-90,'R'),(180,'B')]:
  x,y=endpoint(cx,cy,r+13,a);d.text((x,y),l,font=font(15,True),fill=MUTED,anchor='mm')
def model_input(im,method):
 if method=='single-max':
  w,h=im.size
  side=min(w,h)
  x=(w-side)//2
  y=(h-side)//2
  crop=im.crop((x,y,x+side,y+side))
  return crop.resize((224,224),Image.Resampling.BILINEAR)
 scale=min(224/im.width,224/im.height);sz=tuple(max(1,min(224,round(v*scale))) for v in im.size)
 tile=Image.new('RGB',(224,224),(124,116,104));tile.paste(im.resize(sz,Image.Resampling.BILINEAR),((224-sz[0])//2,(224-sz[1])//2));return tile
for method,label,source in [('single-max','Single Crop Max','single-max')]:
 selected=[r for r in rows if r['method']==source]
 preds={(r['timestamp'],r['camera']):r for r in selected};ts=sorted({r['timestamp'] for r in selected})
 assert len(ts)==200 and len(preds)==len(selected)==1000
 for t in ts:
  for cam in CAMS:
   r=preds[t,cam];meta=json.loads(r['details_json']);p=DATA/meta['path']
   assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['input_sha256'],p
   assert math.isfinite(float(r['sun_deg'])) and math.isfinite(float(r['heading_deg']))
 print(method,'verified 1,000 image hashes',flush=True)
 mp4=OUT/f'scene_006-{source}-five-cameras.mp4'
 writer=imageio_ffmpeg.write_frames(str(mp4),(W,H),fps=5,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','19'])
 writer.send(None)
 frame_ids=list(range(0,len(ts),2));frame_records=[]
 for n,i in enumerate(frame_ids):
  t=ts[i];canvas=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(canvas)
  d.text((20,15),f'SCENE 006  /  {label.upper()}  /  FIVE SYNCHRONIZED CAMERAS',font=font(29,True),fill=FG)
  d.text((20,57),f'Frame {i+1:03d} / 200   |   nominal t = {i/10:04.1f}s   |   source timestamp {t}',font=font(20),fill=MUTED)
  d.text((20,84),'Full views give visual context. Colored arrows show model estimates, not measured solar positions.',font=font(17),fill=MUTED)
  aligned=[]
  for cam,color,(x,y) in zip(CAMS,COLORS,POS):
   r=preds[t,cam];a=float(r['sun_deg']);heading=float(r['heading_deg']);aligned.append(wrap(a+heading))
   d.rounded_rectangle((x,y,x+512,y+468),radius=10,fill=PANEL)
   d.text((x+12,y+10),cam.upper(),font=font(21,True),fill=color)
   with Image.open(DATA/json.loads(r['details_json'])['path']) as f:im=f.convert('RGB')
   photo=im.resize((488,275),Image.Resampling.LANCZOS);canvas.paste(photo,(x+12,y+42))
   if method=='single-max':
    w,h=im.size
    side=min(w,h)
    l=((w-side)//2)*488/w;top=((h-side)//2)*275/h
    d.rectangle((x+12+l,y+42+top,x+12+l+side*488/w,y+42+top+side*275/h),outline=GOLD,width=2)
   inp=model_input(im,method).resize((116,116),Image.Resampling.NEAREST);canvas.paste(inp,(x+12,y+331))
   d.text((x+12,y+449),'Input preview',font=font(14),fill=MUTED)
   cx,cy=x+214,y+389;dial(d,cx,cy,42);arrow(d,cx,cy,40,a,color)
   d.text((x+280,y+335),'MODEL SUN DIRECTION',font=font(15,True),fill=color)
   d.text((x+280,y+359),f'{a:+.1f}°',font=font(30,True),fill=FG)
   d.text((x+280,y+399),'0° forward, +90° left',font=font(16),fill=MUTED)
   d.text((x+280,y+424),'Dial is camera-relative',font=font(16),fill=MUTED)
  x,y=536,588;d.rounded_rectangle((x,y,x+512,y+468),radius=10,fill=PANEL)
  d.text((x+16,y+12),'COMPARE DIRECTIONS',font=font(23,True),fill=FG)
  d.text((x+16,y+43),'Front-camera frame • yaw alignment only',font=font(18),fill=MUTED)
  cx,cy=x+126,y+188;dial(d,cx,cy,82)
  for a,color in zip(aligned,COLORS):arrow(d,cx,cy,77,a,color,3)
  for j,(cam,color,a) in enumerate(zip(CAMS,COLORS,aligned)):
   d.text((x+253,y+100+j*35),f'{cam}: {a:+.0f}°',font=font(18),fill=color)
  d.text((x+16,y+304),'ACTUAL SUN: NOT ANNOTATED',font=font(20,True),fill=GOLD)
  d.text((x+16,y+337),'Use all five views for sun / glare / shadow context.',font=font(17),fill=FG)
  d.text((x+16,y+364),'COLMAP gives camera orientation, not sun truth.',font=font(17),fill=MUTED)
  d.text((x+16,y+392),'Aligned arrows should roughly agree if consistent;',font=font(17),fill=MUTED)
  d.text((x+16,y+418),'camera tilt is ignored. Agreement is not accuracy.',font=font(17),fill=MUTED)
  note='Largest centered square crop (side = min(width,height)), 224×224 preview before normalization; internal transforms bypassed.'
  d.text((20,1070),note,font=font(18),fill=FG)
  d.text((20,1100),'Saved inference replay • every other source frame at 5 fps • 20-second loop • no inferred image-plane sun marker',font=font(18),fill=MUTED)
  writer.send(canvas.tobytes())
  if n in (0,50,99):canvas.save(OUT/f'scene_006-{source}-frame-{i:03}.png')
  frame_records.append({'frame_index':i,'timestamp':t,'aligned_sun_angles':dict(zip(CAMS,aligned))})
 writer.close()
 gif=mp4.with_suffix('.gif');ff=imageio_ffmpeg.get_ffmpeg_exe()
 subprocess.run([ff,'-v','error','-i',str(mp4),'-filter_complex','[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3','-loop','0','-y',str(gif)],check=True)
 with Image.open(gif) as f:
  assert f.n_frames==100 and f.size==(W,H)
  durations=[]
  for k in range(f.n_frames):f.seek(k);durations.append(f.info['duration'])
  assert sum(durations)==20000
 checks.append({'method':source,'source_method':source,'input_hashes_verified':1000,'frames':100,'size':[W,H],'duration_ms':sum(durations),'gif_bytes':gif.stat().st_size,'mp4_bytes':mp4.stat().st_size,'frames_and_angles':frame_records})
 print(source,'finished',flush=True)
(OUT/'validation.json').write_text(json.dumps(checks,indent=2))
