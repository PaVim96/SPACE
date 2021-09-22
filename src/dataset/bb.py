import os
import pandas as pd
import numpy as np
import random
import torch
from torchvision.utils import draw_bounding_boxes as draw_bb
from torchvision.utils import save_image


def _bb_pacman(img_gt):
    enemy_list = ['sue', 'inky', 'pinky', 'blinky']
    pieces = {"save_fruit": (171, 139, "S"), "score0": (183, 86, "S"),
              "pacman": (img_gt['player_y'], img_gt['player_x'], "M"),
              "fruit": (img_gt['fruit_y'], img_gt['fruit_x'], "M")}
    for en in enemy_list:
        if img_gt[f'{en}_visible']:
            pieces[en] = (img_gt[f'{en}_y'], img_gt[f'{en}_x'], "M")
    if img_gt['lives'].item() >= 2:
        pieces["life1"] = (170, 25, "S")
    if img_gt['lives'].item() >= 3:
        pieces["life2"] = (170, 42, "S")
    IMG_SIZE = 128
    return pd.DataFrame.from_dict({
        k: [xy[1] * IMG_SIZE / 160.0 - 11, xy[0] * IMG_SIZE / 210.0,
            0.07 * IMG_SIZE if k != 'score0' else 0.2 * IMG_SIZE, 0.07 * IMG_SIZE, xy[2]]
        for k, xy in pieces.items()
    }, orient='index')


def _bb_carnival(img_gt):
    import ast
    print("Warning: Carnival does not yet mark moving objects")
    img_gt = gt.iloc[[idx]]
    bbs = [(img_gt['bullets_x'].item() * 128 / 160.0 - 2, 198 * 128 / 210.0, 0.04 * 128, 0.04 * 128)]
    for animal in ['shooters', 'rabbits', 'owls', 'ducks', 'flying_ducks']:
        for x, y in ast.literal_eval(img_gt[animal].item()):
            bbs.append((x * 128 / 160.0 - 5, y * 128 / 210.0 - 5, 0.08 * 128, 0.08 * 128))
    for x, y in ast.literal_eval(img_gt['refills'].item()):
        bbs.append((x * 128 / 160.0 - 4, y * 128 / 210.0 - 1, 0.07 * 128, 0.05 * 128))

    if img_gt['bonus'].item():
        bbs.append((12 * 128 / 160.0, 29 * 128 / 210.0, 21 * 128 / 210.0, 8 * 128 / 210.0))

    return pd.DataFrame.from_dict({i: [bb[0], bb[1], bb[2], bb[3]] for i, bb in enumerate(bbs)}, orient='index')


def _bb_pong(img_gt):
    return pd.DataFrame.from_dict({0: [0, 1, 2, 3]}, orient='index')

def save(args, info, output_path):
    if args.game == "MsPacman":
        bb = _bb_pacman(info)
    elif args.game == "Carnival":
        bb = _bb_carnival(info)
    elif args.game == "Pong":
        bb = _bb_pong(info)
    else:
        raise ValueError(f'Unsupported Game supplied: {args.game}')
    bb = bb[(bb[0] > 0) & (bb[0] < 128) & (bb[1] > 0) & (bb[1] < 128)]
    bb.to_csv(output_path, header=False, index=False)
