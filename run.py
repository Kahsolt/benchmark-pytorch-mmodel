#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/02 

from pathlib import Path
from argparse import ArgumentParser
from time import time
from typing import Callable, List

import torch
from torch.nn import Module, ModuleList, Linear, Conv1d
from torch.optim import SGD
import torch.nn.functional as F

BASE_PATH = Path(__file__).parent
TMP_PATH = BASE_PATH / 'tmp' ; TMP_PATH.mkdir(exist_ok=True)


def hug_me(fn:Callable):
  def wrapper(args):
    torch.manual_seed(args.seed)

    import gc; gc.collect()
    if args.device == 'cuda':
      torch.cuda.reset_accumulated_memory_stats()
      torch.cuda.reset_peak_memory_stats()
      torch.cuda.ipc_collect()
      torch.cuda.manual_seed_all(args.seed)

    print(f'>> [{fn.__name__}]')
    t = time()
    r = fn(args)
    t_run = time() - t

    if args.device == 'cuda':
      alloc = torch.cuda.max_memory_allocated() // 2**10
      resrv = torch.cuda.max_memory_reserved()  // 2**10
      print(f'>> vram usage:')
      print(f'     max alloc: {alloc} KB')
      print(f'     max resrv: {resrv} KB')
      torch.cuda.ipc_collect()
    import gc; gc.collect()

    print(f'>> {fn.__name__} done in {t_run:.3f}s')

    return r
  return wrapper

def fake_train(args, getM:Callable, getX:Callable, genY:Callable) -> List[float]:
  t = time()
  model: Module = getM()
  optim = SGD(model.parameters(), lr=0.01, momentum=0.9)
  t_m = time() - t
  param_cnt = sum([p.numel() for p in model.parameters() if p.requires_grad])

  with open(TMP_PATH / f'{args.model}.txt', 'w', encoding='utf-8') as fh:
    fh.write(str(model))

  t_d, t_zg, t_f, t_b, t_o = 0.0, 0.0, 0.0, 0.0, 0.0
  for _ in range(args.count):
    # data transfer
    t = time()
    X = getX()
    Y = genY()
    t_d += time() - t
    
    # clear grad
    t = time()
    optim.zero_grad()
    t_zg += time() - t

    # model forward
    t = time()
    Y_hat = model(X)
    t_f += time() - t

    # model backward
    t = time()
    loss = F.mse_loss(Y_hat, Y)
    loss.backward()
    t_b += time() - t

    # optim step
    t = time()
    optim.step()
    t_o += time() - t

  print(f'>> model define: {t_m:.3f}s')
  print(f'     param_cnt: {param_cnt}')
  print(f'>> data: {t_d:.3f}s')
  print(f'>> zero grad: {t_zg:.3f}s')
  print(f'>> forward: {t_f:.3f}s')
  print(f'>> backward: {t_b:.3f}s')
  print(f'>> optim: {t_o:.3f}s')

  return t_m, t_d, t_zg, t_f, t_b, t_o


def get_expname(args):
  return f'{args.model}_B={args.batch_size}_I={args.d_in}_O={args.d_out}-{args.device}'


class MModel(Module):
  def __init__(self, args):
    super().__init__()
  def forward(self, x):
    return torch.stack([mod(x[i]) for i, mod in enumerate(self.mods)], dim=0)

class MLinear(MModel):
  def __init__(self, args):
    super().__init__(args)
    self.mods = ModuleList([Linear(args.d_in, args.d_out) for _ in range(args.n_model)])

class MConv1d(MModel):
  def __init__(self, args):
    super().__init__(args)
    self.mods = ModuleList([Conv1d(args.d_in, args.d_out, kernel_size=1) for _ in range(args.n_model)])


@hug_me
def go_linear(args):
  getM = lambda: Linear(args.d_in, args.d_out).to(device, dtype)
  genX = lambda: torch.rand([args.batch_size, args.d_in ]).to(device, dtype)
  genY = lambda: torch.rand([args.batch_size, args.d_out]).to(device, dtype)
  return fake_train(args, getM, genX, genY)

@hug_me
def go_conv1d(args):
  getM = lambda: Conv1d(args.d_in, args.d_out, kernel_size=1).to(device, dtype)
  genX = lambda: torch.rand([args.batch_size, args.d_in,  1]).to(device, dtype)
  genY = lambda: torch.rand([args.batch_size, args.d_out, 1]).to(device, dtype)
  return fake_train(args, getM, genX, genY)

@hug_me
def go_mlinear(args):
  getM = lambda: MLinear(args).to(device, dtype)
  genX = lambda: torch.rand([args.n_model, args.batch_size, args.d_in ]).to(device, dtype)
  genY = lambda: torch.rand([args.n_model, args.batch_size, args.d_out]).to(device, dtype)
  return fake_train(args, getM, genX, genY)

@hug_me
def go_mconv1d(args):
  getM = lambda: MConv1d(args).to(device, dtype)
  genX = lambda: torch.rand([args.n_model, args.batch_size, args.d_in,  1]).to(device, dtype)
  genY = lambda: torch.rand([args.n_model, args.batch_size, args.d_out, 1]).to(device, dtype)
  return fake_train(args, getM, genX, genY)

@hug_me
def go_gconv1d(args):
  getM = lambda: Conv1d(args.n_model*args.d_in, args.n_model*args.d_out, kernel_size=1, groups=args.n_model).to(device, dtype)
  genX = lambda: torch.rand([args.batch_size, args.n_model*args.d_in,  1]).to(device, dtype)
  genY = lambda: torch.rand([args.batch_size, args.n_model*args.d_out, 1]).to(device, dtype)
  return fake_train(args, getM, genX, genY)


if __name__ == '__main__':
  runners = {k: v for k,v in globals().items() if k.startswith('go_')}
  SMODELS = ['linear', 'conv1d']
  MMODELS = ['mlinear', 'mconv1d', 'gconv1d']
  MODELS = SMODELS + MMODELS

  parser = ArgumentParser()
  parser.add_argument('-D', '--device',     default='cuda', choices=['cpu', 'cuda'])
  parser.add_argument('-T', '--dtype',      default='float32', choices=['float32', 'float16', 'bfloat16'])
  parser.add_argument('-B', '--batch_size', default=32,  type=int)
  parser.add_argument('-I', '--d_in',       default=256, type=int)
  parser.add_argument('-O', '--d_out',      default=512, type=int)
  parser.add_argument('-K', '--n_model',    default=-1,  type=int, help='if > 0, run multi-models for parallel test')
  parser.add_argument('-M', '--model', choices=MODELS, help='only run the given model')
  parser.add_argument('--count', default=10000,  type=int, help='repeat fake train times')
  parser.add_argument('--seed',  default=114514, type=int)
  args = parser.parse_args()

  dtype: torch.dtype = getattr(torch, args.dtype)
  device = args.device
  if device == 'cuda':
    assert torch.cuda.is_available()

  if args.model:
    runners[f'go_{args.model}'](args)
    exit(0)

  names = MMODELS if args.n_model > 0 else SMODELS
  for name in names:
    args.model = name
    ts = runners[f'go_{name}'](args)
    print('==============================')
