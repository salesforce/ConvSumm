'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import numpy as np
import torch
import os


def compute_accuracy(prediction, gtruth):
    assert len(prediction) == len(gtruth)
    correct = np.sum(np.array(prediction) == np.array(gtruth))
    return correct / len(prediction)


def save_model(model, output_dir, ep_num):
    model_to_save = (
        model.module if hasattr(model, 'module') else model
    )
    model_name = 'model_' + str(ep_num) + '.bin'
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, model_name))
