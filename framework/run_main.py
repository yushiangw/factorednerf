import os
import argparse 
from src import config


def get_parser():

    parser = argparse.ArgumentParser(  )
    
    parser.add_argument('-f','--config', type=str, required=True, 
                                help='Path to config file.')
    
    parser.add_argument('--mode', type=str, required=True)
    
    parser.add_argument('--stage', default=None )
    
    parser.add_argument('--debug', action='store_true', default=False, help='')

    parser.add_argument('--no_dir', action='store_true', default=False )

    parser.add_argument('--skip_train', action='store_true', default=False, help='')
    parser.add_argument('--add_ts', 
                action='store_true', default=False, help='')

    parser.add_argument('-c', '--cont', type=str, default=None, help='')



    parser.add_argument('--rdm', type=str, default=None, help='render mode [obj,all]')

    parser.add_argument('--fnum', type=int, default=None, help='frame number')
    parser.add_argument(  '--dw', type=int, default=None, help='down sample ratio in [1,2,4,]')

    parser.add_mutually_exclusive_group(required=False)

    return parser 

def get_entry(*vargs):

    parser = get_parser()

    args = parser.parse_args(*vargs)

    cfg = config.load_config(args.config)

    cfg_name = os.path.basename(args.config).replace('.yaml','')
    
    script_name = cfg['script_name']

    _mod   = __import__(script_name, fromlist=['Helper'])
    Helper  = getattr(_mod, 'Helper') 

    no_dir =  args.mode !='train'

    h = Helper(cfg, args, cfg_name, no_dir=no_dir)

    return h,cfg,args

def run():

    h,cfg,args = get_entry()

    if args.mode =='train':
        h.run_train()
    
    elif args.mode =='train_stage_x':
        h.run_train(stage=args.stage)

    elif args.mode =='render_train':
        h.run_render(mode='train', fnum=args.fnum, 
                    downsample=args.dw)
        
    elif args.mode =='render_valid':
        h.run_render(mode='valid', fnum=args.fnum, 
                    downsample=args.dw)
            
    elif args.mode =='render_train_q':
        h.run_render(mode='train', fnum=args.fnum, steps=10, 
                    downsample=args.dw)
        
    elif args.mode =='render_valid_q':
        h.run_render(mode='valid', fnum=args.fnum, steps=10, 
                    downsample=args.dw)
        
    elif args.mode =='eval':
        h.run_eval(run_train=True, run_val=True)

    elif args.mode =='eval_train':

        h.run_eval(run_train=True, run_val=False)
        
    elif args.mode =='eval_val':

        h.run_eval(run_train=False, run_val=True)
    else:
        raise Exception('error mode:'+args.mode)


if __name__ == '__main__':

    run()
    print('==================')
    print('all done')
    print('==================')