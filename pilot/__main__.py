import argparse
import json
from pathlib import Path
from .core import dumps

def main():
    p=argparse.ArgumentParser(description='Sun Compass / WayveScenes101 pilot')
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('prepare','validate'):
        q=sub.add_parser(cmd);q.add_argument('--data',required=True)
        q.add_argument('--scene-id',default='scene_006')
    for cmd in ('run','benchmark'):
        q=sub.add_parser(cmd)
        for name in ('data','database','weights','run-id'):q.add_argument('--'+name,required=True)
        q.add_argument('--method',choices=['single','scan','letterbox','single-max'],required=True)
        q.add_argument('--subset',choices=['all','benchmark','recovery'],default='benchmark' if cmd=='benchmark' else 'all')
        q.add_argument('--fault-after-predictions',type=int,default=0,help='Disposable recovery tests only: SIGKILL after N uncommitted/committed prediction attempts')
    q=sub.add_parser('status');q.add_argument('--database',required=True)
    q=sub.add_parser('report');q.add_argument('--database',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('backup');q.add_argument('--database',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('restore-check');q.add_argument('--backup',required=True);q.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command in ('prepare','validate'):
        from .prepare import prepare,validate
        (prepare if args.command=='prepare' else validate)(args.data,args.scene_id)
    elif args.command in ('run','benchmark'):
        from .runner import execute
        execute(args)
    else:
        from .report import summarize,report,backup,restore_check
        if args.command=='status': result=summarize(args.database)
        elif args.command=='report': result=report(args.database,args.output)
        elif args.command=='backup': result=backup(args.database,args.output)
        else: result=restore_check(args.backup,args.output)
        print(dumps(result))

if __name__=='__main__':main()
