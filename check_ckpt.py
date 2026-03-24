import sys, torch
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print('Keys:', sorted(ckpt.keys()))
if 'loops' in ckpt: print('Loops:', list(ckpt['loops'].keys()))
if 'global_step' in ckpt: print('Global step:', ckpt['global_step'])
if 'optimizer_states' in ckpt: print('Optimizer saved: yes')
if 'lr_schedulers' in ckpt: print('LR scheduler saved: yes')
